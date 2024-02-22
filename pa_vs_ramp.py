#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_clean_3.csv')
# Convert types
import ast
def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s  # Return original value if conversion fails
df['local_date'] = pd.to_datetime(df['local_date'])
df['tuples_raw'] = df['tuples_raw'].apply(string_to_list)
df['p0_timestamps'] = df['p0_timestamps'].apply(string_to_list)
df['useless_p0_timestamps'] = df['useless_p0_timestamps'].apply(string_to_list)
print(df.info())
#%%
df_ramp = pd.read_csv('df_ramp_local_date.csv')
df_ramp['local_date'] = pd.to_datetime(df_ramp['local_date'])
print(df_ramp.info())
#%%
df_CP = pd.read_csv('df_CP.csv')
df_CP['local_date'] = pd.to_datetime(df_CP['local_date'])
print(df_CP.info())
#%% merge with ramp file and CP file
df = pd.merge(df, df_ramp, on=['user_id', 'local_date'], how='left')
df = pd.merge(df, df_CP, on=['user_id', 'local_date'], how='left')
print(df.info())
#%%
# sorting:
df = df.sort_values(by=['user_id', 'local_date'])
import ast
df['date'] = df['local_date'].dt.strftime('%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'])
#%%
# consider only rows from first row with tuples_raw = list
def keep_rows(group, direction='forward'):
    if direction == 'forward':
        first_list_index = group[group['tuples_raw'].apply(type) == list].index.min()
        if first_list_index is not np.nan:  # If a list type was found in the group
            return group.loc[first_list_index:]
    elif direction == 'backward':
        last_list_index = group[group['tuples_raw'].apply(type) == list].index.max()
        if last_list_index is not np.nan:  # If a list type was found in the group
            return group.loc[:last_list_index]
    return pd.DataFrame()  # Return empty dataframe if no list type was found in the group
#%%
df = df.groupby('user_id', group_keys=False).apply(keep_rows, direction='forward').reset_index(drop=True)
df = df.groupby('user_id', group_keys=False).apply(keep_rows, direction='backward').reset_index(drop=True)
#%%
print(df.info())
#%%
df_ap = df[df['tuples_raw'].apply(lambda x: isinstance(x, list))]
#%%
# count the number of user id:
user_count = df_ap['user_id'].nunique()
print(user_count)
#%%
# select based on artifact percentage
df_ap = df_ap[df_ap['artifact_percentage'] < 5][['user_id', 'date', 'artifact_percentage', 'tuples_raw', 'p0_timestamps', 'local_date']]
user_count = df_ap['user_id'].nunique()
print(user_count)
print(df_ap.info())
# define users id list once and for all
all_users_ids = df_ap['user_id'].unique().tolist()
print(all_users_ids)
#%%
pct_threshold = 50 # in %
a_threshold = 1

def select_tuples(tuples_list, p0_times, time_start=None, time_end=None, effective_duration=None, excluding_p0=False, alpha_threshold=None):

    if not isinstance(tuples_list, list) or not tuples_list:
        return np.nan, np.nan

    # Check if we need to enforce the interval
    if time_start is not None:
        if tuples_list[-1][0] < time_start:
            return np.nan, np.nan
        tuples_list = [t for t in tuples_list if t[0] >= time_start]
        p0_times = [list(filter(lambda x: x >= time_start, sublist)) for sublist in p0_times]
        p0_times = [lst for lst in p0_times if lst]  # To remove empty sublists if any
    if time_end is not None:
        if tuples_list[0][0] > time_end:
            return np.nan, np.nan
        tuples_list = [t for t in tuples_list if t[0] <= time_end]
        p0_times = [list(filter(lambda x: x <= time_end, sublist)) for sublist in p0_times]
        p0_times = [lst for lst in p0_times if lst]  # To remove empty sublists if any

    time_in = tuples_list[0][0]
    time_fin = tuples_list[-1][0]

    # If we're excluding entries where power (the second value) is 0
    if excluding_p0:
        tuples_list = [t for t in tuples_list if t[1] != 0]

    if not tuples_list:
        return np.nan, np.nan

    pct_below_threshold = np.nan
    if alpha_threshold:
        # Count how many values in tuples_list have a third element < alpha_threshold
        count_below_threshold = sum(1 for t in tuples_list if t[2] < alpha_threshold)
        pct_below_threshold = count_below_threshold/len(tuples_list)
        # count_above_1 = sum(1 for t in tuples_list if t[2] > 1)
        # pct_above_1 = count_above_1/len(tuples_list)
        # If less than % values are below the threshold, return np.nan
        if pct_below_threshold < pct_threshold/100:
            return np.nan, np.nan

    # otherwise, if not empty and effective_duration is specified:
    if effective_duration and isinstance(p0_times, list) and p0_times:
        interval_time = time_fin - time_in
        total_p0_time = sum(lst[-1] - lst[0] for lst in p0_times if lst)
        spinning_time = interval_time - total_p0_time
        if spinning_time < 0:
            print('Issue: Spinning time is less than 0')
        if spinning_time < effective_duration:
            return np.nan, np.nan

    return pct_below_threshold, tuples_list
#%%
df_ap['temp_result'] = df_ap.apply(
    lambda row: select_tuples(
        tuples_list=row['tuples_raw'],
        p0_times=row['p0_timestamps'],
        time_start=5 * 60,
        time_end=20 * 60,
        # effective_duration=12 * 60,
        excluding_p0=True #,
        # alpha_threshold=a_threshold
    ), axis=1
)
# df_ap['%_below_a_threshold'] = df_ap['temp_result'].apply(lambda x: x[0])
df_ap['tuples_edit'] = df_ap['temp_result'].apply(lambda x: x[1])
df_ap.drop('temp_result', axis=1, inplace=True)
# Throw all rows with 'tuples_edit' that is not a non-empty list
df_ap = df_ap[df_ap['tuples_edit'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
print(df_ap.info())
#%%
def get_clusters(user_df, min_workouts):
    # Sort the data by date
    user_df = user_df.sort_values('local_date')

    clusters = []
    start_idx = 0
    end_idx = start_idx
    while start_idx < len(user_df):
        end_date = user_df.iloc[start_idx]['date'] + pd.Timedelta(days=cluster_days-1)
        while (end_idx < len(user_df) - 1) and (user_df.iloc[end_idx + 1]['date'] <= end_date):
            end_idx += 1

        cluster = user_df.iloc[start_idx:end_idx + 1]
        if len(cluster) >= min_workouts:
            clusters.append(cluster)

        start_idx = end_idx + 1
        end_idx = start_idx

    return clusters
#%%
pct_dyn_range = 0.20 # 40% Minimum fraction of data in dyn range to consider a group
cluster_days = 30
min_workouts = 10 # to consider a chunk


from scipy.optimize import curve_fit
from scipy.stats import spearmanr
# Initialize an empty DataFrame with specified columns
df_res = pd.DataFrame(columns=['user_id', 'local_date', 'cluster', 'a_dyn_pct', 'cluster_corr_tot', 'cluster_pvalue_corr_tot', 'cluster_corr_avg', 'cluster_pvalue_corr_avg', 'cluster_corr_avg_new', 'cluster_pvalue_corr_avg_new', 'linear_fit_red', 'goodness_linear_fit_red', 'hyperbolic_fit_red', 'goodness_hyperbolic_fit_red', 'linear_fit_full', 'goodness_linear_fit_full', 'hyperbolic_fit_full', 'goodness_hyperbolic_fit_full'])

users_to_plot = []
users_clusters = []
results = []  # Step 1: Initialize an empty list
total_useful_clusters = 0

for user_id in all_users_ids:
    user_data = df_ap[df_ap['user_id'] == user_id]
    user_n = all_users_ids.index(user_id) + 1

    clusters = get_clusters(user_data, min_workouts)
    print(f'User ID {user_id}, clusters {len(clusters)}')

    if len(clusters) > 0:
        # if len(clusters) <= 41:
        users_to_plot.append(user_id)

    for cluster_number, cluster in enumerate(clusters, 1):
        # Defining start_date and end_date
        start_date = cluster['date'].min().strftime("%d-%m-%Y")
        end_date = cluster['date'].max().strftime("%d-%m-%Y")
        end_local_date = cluster['local_date'].max()
        all_tuples = [t for sublist in cluster['tuples_edit'].tolist() for t in sublist]

        # Check if at least 50% of the tuples have third entry < 1
        valid_tuples = [t for t in all_tuples if t[2] < 1]
        percentage_a_dynamical = len(valid_tuples)/len(all_tuples)

        if len(valid_tuples) < pct_dyn_range * len(all_tuples):
            continue  # Skip to next chunk

        total_useful_clusters += 1
        users_clusters.append(user_id)

        p_list = [t[1] for t in all_tuples]
        a_list = [t[2] for t in all_tuples]

        # total correlation
        corr_tot, pvalue_corr_tot = spearmanr(p_list, a_list)

        # correlation in dynamic range
        dynamic_range = [(p, a) for p, a in zip(p_list, a_list) if a <= 1]
        p_list_dyn = [t[0] for t in dynamic_range]
        a_list_dyn = [t[1] for t in dynamic_range]
        corr_dyn, pvalue_corr_dyn = spearmanr(p_list_dyn, a_list_dyn)

        # Determine intervals
        n_intervals = 9
        a_min1 = 0.95 * min(a for a in a_list if a != 0)
        a_max1 = 1.05 * min(max(a_list), 1)
        interval1 = (a_max1 -  a_min1) / n_intervals
        r1 = range(n_intervals)

        average_a_list = []
        average_p_list = []
        for i in r1:
            a_start1 = a_min1 + i * interval1
            a_end1 = a_start1 + interval1
            interval_points1 = [(p, a) for p, a in zip(p_list, a_list) if a_start1 < a < a_end1]

            if len(interval_points1) >= 10:
                average_p = sum(p for p, _ in interval_points1) / len(interval_points1)
                average_a = sum(a for _, a in interval_points1) / len(interval_points1)
                average_a_list.append(average_a)
                average_p_list.append(average_p)

        if len(average_a_list) > 2:
            corr_avg, pvalue_corr_avg = spearmanr(average_p_list, average_a_list)
        else:
            corr_avg, pvalue_corr_avg = np.nan, np.nan

        # try fitting if at least 4 red points in dyn range
        corr_avg_new, pvalue_corr_avg_new = np.nan, np.nan
        m_red, q_red, r2_lin_red = np.nan, np.nan, np.nan
        s_red, t_red, r2_h2_red = np.nan, np.nan, np.nan
        m_full, q_full, r2_lin_full = np.nan, np.nan, np.nan
        s_full, t_full, r2_h2_full = np.nan, np.nan, np.nan
        if len(average_a_list) > 3:
            # collect the rest of interval points and add them to the datapoints to fit if they are at least 2
            # Determine intervals
            n_intervals = 5
            a_min2 = 0.99 * a_end1
            a_max2 = 1.05 * min(max(a_list), 1.8)
            interval2 = (a_max2 -  a_min2) / n_intervals
            r2 = range(n_intervals)

            rest_average_a_list = []
            rest_average_p_list = []
            # Calculate and plot interval centers
            for i in r2:
                a_start2 = a_min2 + i * interval2
                a_end2 = a_start2 + interval2
                interval_points2 = [(p, a) for p, a in zip(p_list, a_list) if a_start2 < a < a_end2]

                if len(interval_points2) >= 10:
                    rest_average_p = sum(p for p, _ in interval_points2) / len(interval_points2)
                    rest_average_a = sum(a for _, a in interval_points2) / len(interval_points2)
                    rest_average_a_list.append(rest_average_a)
                    rest_average_p_list.append(rest_average_p)

            if len(rest_average_p_list) >= 2:
                all_average_a_list = average_a_list + rest_average_a_list
                all_average_p_list = average_p_list + rest_average_p_list

                corr_avg_new, pvalue_corr_avg_new = spearmanr(all_average_p_list, all_average_a_list)

# ------------------------- FITS ------------------------- :
# --------------------------------------------------------
            info_fits = []

            a_red = np.array(average_a_list)
            p_red = np.array(average_p_list)

# 1. Linear regression
            m_red, q_red = np.polyfit(a_red, p_red, 1)
            predicted_linear_red = m_red*a_red + q_red
            r2_lin_red = 1 - (np.sum((p_red - predicted_linear_red) ** 2) / ((len(p_red) - 1) * np.var(p_red, ddof=1)))
            info_fits.append(['linear_dyn', r2_lin_red, [m_red, q_red]])

# 2. Hyperbola p = s/a + t
            def hyperbola2(a, s, t):
                return s/a + t
            params2_red, _ = curve_fit(hyperbola2, a_red, p_red)
            s_red, t_red = params2_red
            predicted_h2_red = hyperbola2(a_red, *params2_red)
            r2_h2_red = 1 - (np.sum((p_red - predicted_h2_red) ** 2) / ((len(p_red) - 1) * np.var(p_red, ddof=1)))
            info_fits.append(['hyper_dyn', r2_h2_red, [s_red, t_red]])

# fit total data only if

            if len(rest_average_p_list) >= 2:
                a_full = np.array(all_average_a_list)
                p_full = np.array(all_average_p_list)

    # 1. Linear regression
                m_full, q_full = np.polyfit(a_full, p_full, 1)
                predicted_linear_full = m_full*a_full + q_full
                r2_lin_full = 1 - (np.sum((p_full - predicted_linear_full) ** 2) / ((len(p_full) - 1) * np.var(p_full, ddof=1)))
                info_fits.append(['linear_full', r2_lin_full, [m_full, q_full]])

    # 2. Hyperbola p = s/a + t
                def hyperbola2(a, s, t):
                    return s/a + t
                params2_full, _ = curve_fit(hyperbola2, a_full, p_full)
                s_full, t_full = params2_full
                predicted_h2_full = hyperbola2(a_full, *params2_full)
                r2_h2_full = 1 - (np.sum((p_full - predicted_h2_full) ** 2) / ((len(p_full) - 1) * np.var(p_full, ddof=1)))
                info_fits.append(['hyper_full', r2_h2_full, [s_full, t_full]])

            if info_fits:
                best_fit_info = max(info_fits, key=lambda x: x[1])
            else:
                best_fit_info = np.nan  # or some default value


        # Create a dictionary with the necessary information
        res_dict = {
            'user_id': user_id,
            'local_date': end_local_date,
            'cluster': cluster_number,
            'a_dyn_pct': percentage_a_dynamical,
            'cluster_corr_tot': corr_tot,
            'cluster_pvalue_corr_tot': pvalue_corr_tot,
            'cluster_corr_avg': corr_avg,
            'cluster_pvalue_corr_avg': pvalue_corr_avg,
            'cluster_corr_avg_new': corr_avg_new,
            'cluster_pvalue_corr_avg_new': pvalue_corr_avg_new,
            'linear_fit_red': [m_red, q_red],
            'goodness_linear_fit_red': r2_lin_red,
            'hyperbolic_fit_red': [s_red, t_red],
            'goodness_hyperbolic_fit_red': r2_h2_red,
            'linear_fit_full': [m_full, q_full],
            'goodness_linear_fit_full': r2_lin_full,
            'hyperbolic_fit_full': [s_full, t_full],
            'goodness_hyperbolic_fit_full': r2_h2_full,
            'best_fit': best_fit_info
        }
        # Append the dictionary to df_res
        results.append(res_dict)

# PLOT IF:

        if user_id in users_to_plot:
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(a_list, p_list , color='lightblue', alpha=0.8, label=f'{len(cluster)} workouts from {start_date} to {end_date}')

            # plot the red points
            ax.scatter(average_a_list, average_p_list, color='red', s=50)

            # plot yellow intervals
            for i in r1:
                a_start1 = a_min1 + i * interval1
                a_end1 = a_start1 + interval1
                ax.axvline(x=a_start1, color='orange', linestyle='--', alpha=0.8)
                if i == r1[-1]:  # For the last loop, add the end boundary too.
                    ax.axvline(x=a_end1, color='orange', alpha=1)

            # plot purple intervals and points if the purple list not empty
            if len(rest_average_a_list) > 0:
                ax.scatter(rest_average_a_list, rest_average_p_list, color='purple', s=50)
                for i in r2:
                    a_start2 = a_min2 + i * interval2
                    a_end2 = a_start2 + interval2
                    ax.axvline(x=a_start2, color='purple', linestyle='--', alpha=0.5)
                    if i == r2[-1]:  # For the last loop, add the end boundary too.
                        ax.axvline(x=a_end2, color='purple', alpha=1)

            # plot BEST linear and hyperbolic fits
            if not np.isnan(r2_lin_red) and (np.isnan(r2_lin_full) or r2_lin_red > r2_lin_full):
                # For smooth plotting of the fit
                a_fine = np.linspace(min(a_red), max(a_red), 400)
                ax.plot(a_fine, m_red*a_fine + q_red, color='red', label=f'Linear: p = {m_red:.2f}*a + {q_red:.2f}, $R^2$ = {r2_lin_red:.2f}')
            if not np.isnan(r2_lin_full) and (np.isnan(r2_lin_red) or r2_lin_full > r2_lin_red):
                # For smooth plotting of the fit
                a_fine = np.linspace(min(a_full), max(a_full), 400)
                ax.plot(a_fine, m_full*a_fine + q_full, color='red', label=f'Linear: p = {m_full:.2f}*a + {q_full:.2f}, $R^2$ = {r2_lin_full:.2f}')
            if not np.isnan(r2_h2_red) and (np.isnan(r2_h2_full) or r2_h2_red > r2_h2_full):
                a_fine = np.linspace(min(a_red), max(a_red), 400)
                ax.plot(a_fine, hyperbola2(a_fine, *params2_red), color='green', label=f'Hyperbola: p = {s_red:.2f}/a + {t_red:.2f}, $R^2$ = {r2_h2_red:.2f}')
            if not np.isnan(r2_h2_full) and (np.isnan(r2_h2_red) or r2_h2_full > r2_h2_red):
                a_fine = np.linspace(min(a_full), max(a_full), 400)
                ax.plot(a_fine, hyperbola2(a_fine, *params2_full), color='green', label=f'Hyperbola: p = {s_full:.2f}/a + {t_full:.2f}, $R^2$ = {r2_h2_full:.2f}')


            ax.set_title(f"User: {user_n}. Corrs: Total {corr_tot:.2f}, Dyn-range {corr_dyn:.2f}, Red: {corr_avg:.2f}, Red+Purple {corr_avg_new:.2f}", fontsize=15)

            ax.set_xlabel("a", fontsize=15)
            ax.set_ylabel("p", fontsize=15)
            ax.tick_params(axis='both', labelsize=12)

            plt.grid(True)
            plt.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig(f'adaptations/{cluster_days}d_group_{user_id}_{cluster_number}.png')
            plt.close()

# This is the end of your loops. Now create the DataFrame from the results list.
df_res = pd.DataFrame(results)
users_clusters = [x for i, x in enumerate(users_clusters) if x not in users_clusters[:i]]
print(f'Total number of plotted (useful) clusters {total_useful_clusters} for {len(users_clusters)} users.')
#%%
print(df_res.info())
print(df_res[['user_id', 'local_date', 'best_fit']].head(5))
#%%
# Here merge df_res to df using user_id and local_date
# (check that other rows are NaNs)

# Perform a left join
df = pd.merge(df, df_res, on=['user_id', 'local_date'], how='left')
#%%
print(df.info())
#%%
# IN CASE OF MESSING UP THE MERGING: DROP
# List of columns to be dropped, excluding 'user_id' and 'local_date'
# cols_to_drop = [col for col in df_res.columns if col not in ['user_id', 'local_date']]
# Drop the columns from df
# df.drop(cols_to_drop, axis=1, inplace=True)
# Keep only the first three columns
df = df.iloc[:, :17]

print(df.info())
# print(df.iloc[1])
#%%
# compute p(0.75) and p(0.5) using the BEST fit

def compute_p_from_best_fit(best_fit_info, a_value):
    if not isinstance(best_fit_info, list):
        return np.nan
    pars = best_fit_info[2]  # third entry contains the best fit parameters
    if 'linear' in best_fit_info[0]:  # Check if the string contains 'linear'
        power = pars[0] * a_value + pars[1]
    else:
        power = pars[0] / a_value + pars[1]
    return power
#%%
df['p_75'] = df.apply(lambda row: compute_p_from_best_fit(row['best_fit'], 0.75), axis=1)
df['p_50'] = df.apply(lambda row: compute_p_from_best_fit(row['best_fit'], 0.5), axis=1)
df['p_1'] = df.apply(lambda row: compute_p_from_best_fit(row['best_fit'], 1), axis=1)
# print(df[df['best_fit'].apply(lambda x: isinstance(x, list))][['best_fit', 'p_50', 'p_75']])
#%%
# check rows
print(df[df['best_fit'].apply(lambda x: isinstance(x, list))][['p_75']])
#%%
# NOTICE WE HAVE TUPLE INFO ONLY IF ALPHA1 IS PRESENT.
# THEREFORE WE LACK HR/POWER INFO IF NO ALPHA1.
# ----> FOCUS ON USERS WITH EXTENSIVE ALPHA1 DATA ONLY AND RELATIVE TIME SPAN
#%%
# baselines for ramp thresholds
# compute date differences and interested baselines
df = df.sort_values(by=['user_id', 'local_date'])
df = df.reset_index(drop=True)
df['date_diff'] = df.groupby('user_id')['date'].diff().dt.days.fillna(0) # safely filling NaN -> 0 because the function applies a check on the user_id!
# %% BASELINE includes today 
def compute_baseline(
        row,
        df,
        column_name,
        duration_days
        ):
    if pd.isna(row[column_name]) or row['date_diff'] == 0:
        return np.nan # if either NaN or not the first workout of the day

    index = row.name  # Get the index of the current row

    days_sum = row['date_diff']
    value_list = []
    if not pd.isna(row[column_name]):
        value_list.append(row[column_name])

    j = index - 1
    while j >= 0 and days_sum <= duration_days:
        if df.loc[j, 'user_id'] != row['user_id']:
            break
        new_date_diff = df.loc[j, 'date_diff']
        new_value = df.loc[j, column_name]
        days_sum += new_date_diff

        # skip this row if it corresponds to the second/third/... workout of the day (i.e, date difference wrt row above is 0)
        if new_date_diff == 0:
            j -= 1
            continue # to the row above

        # we only consider FIRST workout 
        if not pd.isna(new_value):
            value_list.append(new_value)
        j -= 1

    if value_list:
        value_sum = sum(value_list)
        return value_sum / len(value_list)
    else:
        return np.nan  # Return NaN if no values to average

# %%
df['moving_watts_alpha_thr'] = df.apply(
    lambda row: compute_baseline(row, df, 'watts_alpha_thr', cluster_days),
    axis=1
)
df['moving_watts_alpha_end'] = df.apply(
    lambda row: compute_baseline(row, df, 'watts_alpha_end', cluster_days),
    axis=1
)
df['moving_watts_alpha_thr_cluster'] = df.apply(
    lambda row: compute_baseline(row, df, 'watts_alpha_thr_cluster', cluster_days),
    axis=1
)
df['moving_watts_alpha_end_cluster'] = df.apply(
    lambda row: compute_baseline(row, df, 'watts_alpha_end_cluster', cluster_days),
    axis=1
)
df['moving_cp'] = df.apply(
    lambda row: compute_baseline(row, df, 'cp', cluster_days),
    axis=1
)
#%%
print(df.info())
# %%
# Plotting ramp and CP data
for user in all_users_ids:
    user_df = df[df['user_id'] == user]
    user_n = all_users_ids.index(user) + 1

    # if user_df['watts_alpha_thr'].isna().all() and user_df['watts_alpha_end'].isna().all():
    #     continue
    plt.figure(figsize=(10, 6))

    ramp_data_thr = user_df.dropna(subset=['moving_watts_alpha_thr'])[['date', 'moving_watts_alpha_thr']]
    ramp_data_end = user_df.dropna(subset=['moving_watts_alpha_end'])[['date', 'moving_watts_alpha_end']]
    cp_data = user_df.dropna(subset=['moving_cp'])[['date', 'moving_cp']]
    cp_data['moving_cp95'] = 0.95 * cp_data['moving_cp'] 
    cp_data['moving_cp90'] = 0.90 * cp_data['moving_cp'] 

    plt.plot(ramp_data_thr['date'], ramp_data_thr['moving_watts_alpha_thr'], label='moving_watts_alpha_thr', color='blue')
    plt.plot(ramp_data_end['date'], ramp_data_end['moving_watts_alpha_end'], label='moving_watts_alpha_end', color='green')   
    
    
    plt.plot(cp_data['date'], cp_data['moving_cp'], label='moving_cp', color='black', linestyle='--')
    plt.plot(cp_data['date'], cp_data['moving_cp95'], label='moving_cp95', color='blue', linestyle='--')
    plt.plot(cp_data['date'], cp_data['moving_cp90'], label='moving_cp90', color='red', linestyle=':')
    
    
    plt.scatter(user_df['date'], user_df['p_50'], label='p_50', color='purple')
    plt.scatter(user_df['date'], user_df['p_75'], label='p_75', color='red')
    plt.scatter(user_df['date'], user_df['p_1'], label='p_1', color='orange')

    plt.title(f"User {user_n} Ramp vs p(a) Thresholds vs CP ({cluster_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Watts")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{cluster_days}d_pa_ramp_cp_{user}.png')
    plt.close()

#%%
# Plotting ramp cluster
for user in all_users_ids:
    user_df = df[df['user_id'] == user]
    user_n = all_users_ids.index(user) + 1

    if user_df['watts_alpha_thr_cluster'].isna().all() and user_df['watts_alpha_end_cluster'].isna().all():
        continue
    plt.figure(figsize=(10, 6))

    ramp_data_thr = user_df.dropna(subset=['moving_watts_alpha_thr_cluster'])[['date', 'moving_watts_alpha_thr_cluster']]
    ramp_data_end = user_df.dropna(subset=['moving_watts_alpha_end_cluster'])[['date', 'moving_watts_alpha_end_cluster']]
    cp_data = user_df.dropna(subset=['moving_cp'])[['date', 'moving_cp']]


    plt.plot(ramp_data_thr['date'], ramp_data_thr['moving_watts_alpha_thr_cluster'], label='moving_watts_alpha_thr_cluster', color='blue')
    plt.plot(ramp_data_end['date'], ramp_data_end['moving_watts_alpha_end_cluster'], label='moving_watts_alpha_end_cluster', color='green')   
    plt.plot(cp_data['date'], cp_data['moving_cp'], label='moving_cp', color='black', linestyle='--')

    plt.scatter(user_df['date'], user_df['p_50'], label='p_50', color='purple')
    plt.scatter(user_df['date'], user_df['p_75'], label='p_75', color='red')
    plt.scatter(user_df['date'], user_df['p_1'], label='p_1', color='orange')

    plt.title(f"User {user_n} Threshold Powers: p(a) vs ramp_cluster (over {cluster_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Watts")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{cluster_days}d_ramp_clusters_{user}.png')
    plt.close()
# %%
print(df.info())
# %%
