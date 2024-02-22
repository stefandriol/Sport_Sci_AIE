import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_clean2.csv')
#%%
# Convert types
import ast
def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s  # Return original value if conversion fails
df['local_date'] = pd.to_datetime(df['local_date'])
df['tuples_raw'] = df['tuples_raw'].apply(string_to_list)
print(df.info())
#%%
# check all possible types of tuples_raw
unique_types = df['tuples_raw'].apply(type).unique()
# Display the unique types
print(unique_types)
# check all possible types of tuples_raw
unique_types = df['ess'].apply(type).unique()
# Display the unique types
print(unique_types)
unique_types = df['artifact_percentage'].apply(type).unique()
# Display the unique types
print(unique_types)
#%%
df = df.sort_values(by=['user_id', 'local_date'])
# by timestamp
df['tuples_raw'] = df['tuples_raw'].apply(lambda x: sorted(x, key=lambda t: t[0]) if (isinstance(x, list) and x) else None)
#%%
import ast

df['date'] = df['local_date'].dt.strftime('%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'])
#%%
# get rid of all rows with no tuples = list, before the first (and after the last) row where tuple = list

def keep_after_list_found(group):
    first_list_index = group[group['tuples_raw'].apply(type) == list].index.min()
    if first_list_index is not np.nan:  # If a list type was found in the group
        return group.loc[first_list_index:]
    return pd.DataFrame()  # Return empty dataframe if no list type was found in the group

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

# Apply the function forward
df = df.groupby('user_id').apply(keep_rows, direction='forward').reset_index(drop=True)

# Apply the function backward
df = df.groupby('user_id').apply(keep_rows, direction='backward').reset_index(drop=True)

# Reset the multi-index of the result
df = df.reset_index(drop=True)
#%%
print(df.info())
#%%
print(df.head())
#%%
# averages of tuple quantities

def compute_average(tuples_list, option, time_start=None, time_end=None, effective_duration=None, excluding_p0=False):

    if not isinstance(tuples_list, list) or not tuples_list:
        return None
    """
    Compute the average of a specific entry in the tuples.

    Args:
    - tuples_list (list of tuple): List of tuples to be averaged.
    - option (str): Specifies which value in the tuple to average. Can be 'power' (2nd entry of tuple), 'alpha1' (3rd), or 'heartrate' (4th).

    Returns:
    - float: Average of the specified entry.
    """
    # Check if we need to enforce the interval
    if time_start is not None:
        if tuples_list[-1][0] < time_start:
            return None
        tuples_list = [t for t in tuples_list if t[0] >= time_start]
    if time_end is not None:
        if tuples_list[0][0] > time_end:
            return None
        tuples_list = [t for t in tuples_list if t[0] <= time_end]
    # If we're excluding entries where power (the second value) is 0
    if excluding_p0:
        tuples_list = [t for t in tuples_list if t[1] != 0]

    if not tuples_list or (effective_duration and len(tuples_list)*5 < effective_duration):
        return None  # Return None if the list is empty

    if option == 'power':
        return sum(t[1] for t in tuples_list) / len(tuples_list)
    elif option == 'alpha1':
        return sum(t[2] for t in tuples_list) / len(tuples_list)
    elif option == 'hr':
        return sum(t[3] for t in tuples_list) / len(tuples_list)
    else:
        raise ValueError("Option not recognized. Choose from 'power', 'alpha1', or 'hr'.")
#%%
df['avg_power_20'] = df['tuples_raw'].apply(compute_average, option='power', time_start=5*60, time_end=25*60)
df['avg_alpha1_20'] = df['tuples_raw'].apply(compute_average, option='alpha1', time_start=5*60, time_end=25*60)

df['effective_power_20'] = df['tuples_raw'].apply(compute_average, option='power', time_start=5*60, time_end=25*60, effective_duration=15*60, excluding_p0=True)
df['effective_alpha1_20'] = df['tuples_raw'].apply(compute_average, option='alpha1', time_start=5*60, time_end=25*60, effective_duration=15*60, excluding_p0=True)
#%%
print(df.info())
#%%
# consider effective power and SMALLER (avg_alpha, eff_alpha) to construct the p_a as product

# --------------- COMPUTE pa 20 mins ---------------

# p_a: only if eff power is there
df['pa'] = df['effective_power_20'] * np.minimum(df['avg_alpha1_20'], df['effective_alpha1_20']).where(df['effective_power_20'].notna(), np.nan)
#%%
# check which ess make sense: how many standard deviations?

# You can drop the 'mean' and 'std' columns if you don't need them anymore
df.drop(columns=['mean', 'std'], inplace=True)
# Step 1: Get unique ess_tot for each user_id and date
unique_ess = df.groupby(['user_id', 'date'])['ess_tot'].first().reset_index()
# Step 2: Calculate mean and std of ess_tot for each user_id
stats = unique_ess.groupby('user_id')['ess_tot'].agg(['mean', 'std']).reset_index()
# Step 3: Merge stats with the original dataframe
df = df.merge(stats, on='user_id', how='left')
# Step 4: Calculate ess_tot_sd for each row
df['ess_tot_sd'] = (df['ess_tot'] - df['mean']) / df['std']
#%%
print(stats.info())
# You can drop the 'mean' and 'std' columns if you don't need them anymore
# df.drop(columns=['mean_x', 'std_x'], inplace=True)
# df.drop(columns=['mean_y', 'std_y'], inplace=True)
print(df.info())
#%%
# plot histogram

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Unique user_ids in the DataFrame
unique_users = df['user_id'].unique()

for user in unique_users:
    subset = df[df['user_id'] == user]

    # Extract unique 'ess_tot' values for rows with different 'date'
    grouped = subset.groupby('date').first().reset_index()
    values = grouped['ess_tot']

    # Extract Mean and Std
    mu, std = subset['mean'].iloc[0], subset['std'].iloc[0]
    # Now you can use 'values', 'mu', and 'std' for further operations

    # Plotting the histogram
    plt.hist(values, bins=30, density=True, alpha=0.6, color='g', label='Observed Distribution')

    # Plotting the Gaussian
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fitted Gaussian')
    plt.title(f"Fit results: mu = {mu:.2f}, std = {std:.2f}")

    # Labeling the plot
    plt.xlabel('ess_tot')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Normalized Frequency and Gaussian fit for user {user}')

    # Saving the plot
    filename = f'ess_distribution_user_{user}.png'
    plt.savefig(filename)

    # Clearing the current figure to avoid overlap in the next iteration
    plt.clf()
#%%
# delete rows with weird ESS > 600 (400-600 may make sense for a race)

weird_ess = df[df['ess_tot']>600]
print(len(weird_ess))

print(weird_ess[['user_id', 'date', 'pa', 'ess', 'ess_tot']])

# remove them
df = df[df['ess_tot'] <= 600]
#%%
# define day date to define date difference as well as baseline

df['date_diff'] = df.groupby('user_id')['date'].diff().dt.days.fillna(0) # safely filling NaN -> 0 because the function applies a check on the user_id!
#%%
# construct ess_tot

# Group by 'user_id' and 'local_date' and aggregate 'ess' values by summing them

# Step 1: Aggregate on 'user_id' and 'local_date' to compute the total 'ess' for each day.
# Group by 'user_id' and 'date' to get counts for each combination.
date_counts = df.groupby(['user_id', 'date']).size().reset_index(name='counts')
# Sort by 'user_id' and 'counts' in descending order
date_counts_sorted = date_counts.sort_values(by=['user_id', 'counts'], ascending=[True, False])
# Group by 'user_id' and pick the first row which will have the maximum count
max_date_counts = date_counts_sorted.groupby('user_id').first().reset_index()
print(max_date_counts)

print(df[(df['date'] == '2022-07-05') & (df['user_id'] == 975)])
#%%
# issue: many rows for same date :/

df.drop(columns=['ess_tot'], inplace=True)
# Group by user_id and date to calculate the sum of ess
agg_df = df.groupby(['user_id', 'date'])['ess'].sum().reset_index()
agg_df = agg_df.rename(columns={'ess': 'ess_tot'})

# Merge the aggregated data with the original dataframe.
df = pd.merge(df, agg_df, on=['user_id', 'date'], how='left')
#%%
print(agg_df.info())
#%%
def compute_baseline(
        row,
        df,
        column_name,
        duration_days,
        min_appearances,
        # including_today=False, # MODIFY function in case you want to consider it
        art_pct=None
        ):
    if pd.isna(row[column_name]):
        return None

    index = row.name  # Get the index of the current row

    if index < min_appearances - 1:
        return None

    days_sum = row['date_diff']
    value_sum = 0
    dates_set = set()

    # if including_today: # also consider artifact condition
    #    value_sum = row[column_name]
    #    dates_set.add(row['date'])

    j = index - 1
    while j >= 0 and days_sum <= duration_days:
        if not df.loc[j, 'user_id'] == row['user_id']:
            break
        new_date = df.loc[j, 'date']
        new_date_diff = df.loc[j, 'date_diff']
        new_value = df.loc[j, column_name]
        if art_pct:
            artifact_value = df.loc[j, 'artifact_percentage']
            if not pd.isna(artifact_value):
                artifact_condition = artifact_value < art_pct
            else:
                artifact_condition = False
        else:
            artifact_condition = True


        days_sum += new_date_diff

        # skip this row if has same date as starting row or if it corresponds to the second/third/... workout of the day (i.e, date difference wrt row above is 0)
        if new_date == row['date'] or new_date_diff == 0:
            j -= 1
            continue

        # The following executes for rows that are not the main one, and contain info on the first workout of the day:
        # we only consider FIRST workout AND with non-NaN value

        # However, in case the first workout(s) has value == NaN, un-comment all the following if you want the sum to be constructed with the first meaningful value after that
        if not pd.isna(new_value) and artifact_condition:
            # if new_date not in dates_set:
                dates_set.add(new_date)
                value_sum += new_value
            # else:
                # Logic to update value_sum if new_date is already in dates_set
                # to_subtract = None
                # for k in range(j+1, min(j+7, len(df))):  # min() ensures not to exceed the df length
                #    potential_value = df.loc[k, column_name]
                #    if not pd.isna(potential_value):
                #        to_subtract = potential_value
                #        break

                # if to_subtract is not None:  # ensure we have a valid value to subtract
                #    value_sum = value_sum - to_subtract + new_value
                # else:
                #    print('Issue: to_subtract is None')
        if days_sum > duration_days:
                break
        j -= 1

    if len(dates_set) >= min_appearances:
        return value_sum / len(dates_set)
    else:
        return None


df['pa_base_30'] = df.apply(
    compute_baseline,
    args=(df, 'pa', 30, 1),
    axis=1
)

df['ess_base_30'] = df.apply(
    compute_baseline,
    args=(df, 'ess_tot', 30, 1),
    axis=1
)

df['pa_base_7'] = df.apply(
    compute_baseline,
    args=(df, 'pa', 7, 1),
    axis=1
)

df['ess_base_7'] = df.apply(
    compute_baseline,
    args=(df, 'ess_tot', 7, 1),
    axis=1
)
#%%
# baseline ON FIRST WORKOUT of the day:
# check that baselines for different activities on same date are the same

duplicates = df[df.duplicated(subset=['user_id', 'date'], keep=False)]
print(duplicates[['user_id', 'date', 'pa_base_7', 'pa_base_30', 'ess_base_7', 'ess_base_30']])

# construct deviation from baselines only for fisr workout: only if date_diff is NOT 0
# print(df.info())
#%%
# plot points for data, lines for base7 and lines for base30

import matplotlib.pyplot as plt


# Group by user_id and plot data for each user
for user, group in df.groupby('user_id'):

    # Filter rows where pa and ess_tot are available

    ess_base30_data = group.dropna(subset=['ess_base_30'])[['date', 'ess_base_30']]
    pa_base30_data = group.dropna(subset=['pa_base_30'])[['date', 'pa_base_30']]

    ess_base7_data = group.dropna(subset=['ess_base_7'])[['date', 'ess_base_7']]
    pa_base7_data = group.dropna(subset=['pa_base_7'])[['date', 'pa_base_7']]

    pa_data = group.dropna(subset=['pa'])[['date', 'pa']]
    ess_tot_data = group.dropna(subset=['ess_tot'])[['date', 'ess_tot']]

    # Plotting pa data
    plt.scatter(pa_data['date'], pa_data['pa'], marker='o', label='pa', color='lightblue')
    plt.plot(pa_base7_data['date'], pa_base7_data['pa_base_7'], label='pa weekly baseline', color='cornflowerblue')
    plt.plot(pa_base30_data['date'], pa_base30_data['pa_base_30'], label='pa monthly baseline', color='darkslateblue')

    # Plotting ess_tot data
    plt.scatter(ess_tot_data['date'], ess_tot_data['ess_tot'], marker='x', label='ess', color='orange')
    plt.plot(ess_base7_data['date'], ess_base7_data['ess_base_7'], label='ess weekly baseline', color='lightcoral')
    plt.plot(ess_base30_data['date'], ess_base30_data['ess_base_30'], label='ess monthly baseline', color='darkred')

    plt.title(f'pa and daily ess for user: {user}')
    # Adjust x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.ylim(0, 400)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'pa_ess_{user}.png')
    plt.close()
#%%
# construct deviations only for date_diff NOT zero, otherwise considered as MULTIPLE points with same values in the plot and in the correlation computation!

mask0 = ~df['date_diff'].isna() & (df['date_diff'] != 0)
mask = mask0 & ~df['ess_base_7'].isna() & (df['ess_base_7'] != 0) & ~df['ess_base_30'].isna() & (df['ess_base_30'] != 0)  # Not NaN and not zero
df['ess_dev'] = np.where(mask, ((df['ess_base_7'] - df['ess_base_30']) / df['ess_base_30']) * 100, None)
mask = mask0 & ~df['pa_base_7'].isna() & (df['pa_base_7'] != 0) & ~df['pa_base_30'].isna() & (df['pa_base_30'] != 0)  # Not NaN and not zero  # Not NaN and not zero
df['pa_dev'] = np.where(mask, ((df['pa_base_7'] - df['pa_base_30']) / df['pa_base_30']) * 100, None)
#%%
# check:

duplicates = df[df.duplicated(subset=['user_id', 'date'], keep=False)]
print(duplicates[['user_id', 'date', 'pa_dev', 'ess_dev']].head(51))
#%%
# Plot deviations
for user, group in df.groupby('user_id'):

    # Filter rows where pa and ess_tot are available

    ess_dev_data = group.dropna(subset=['ess_dev'])[['date', 'ess_dev']]
    pa_dev_data = group.dropna(subset=['pa_dev'])[['date', 'pa_dev']]

    plt.plot(pa_dev_data['date'], pa_dev_data['pa_dev'], label='pa deviation', color='cornflowerblue')
    plt.plot(ess_dev_data['date'], ess_dev_data['ess_dev'], label='ess deviation', color='lightcoral')

    plt.title(f'Deviations of weekly pa and ess baselines from monthly baselines: {user}')
    # Adjust x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Date')
    plt.ylabel('7-days deviation from 30-days')
    # plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'deviations_{user}.png')
    plt.close()
#%%
# deviations correlation

import matplotlib.pyplot as plt
import numpy as np

correlation_df = df[
    (~df['ess_dev'].isna())
    & (~df['pa_dev'].isna())
    # & (df['ess_dev'] > 10)
]

# Threshold for correlation
X = 20


for user_id in correlation_df['user_id'].unique():
    user_data = correlation_df[correlation_df['user_id'] == user_id]

    x = user_data['ess_dev'].values
    y = user_data['pa_dev'].values

    # Ensure have more than one entry to plot
    if len(x) > 1 :
        plt.figure(figsize=(10, 6))

        plt.axvspan(X, max(x), color='yellow', zorder=0) # zorder sets the stacking order
        # Add vertical dashed black line at x=10
        plt.axvline(x=X, color='black', linestyle='--')

        plt.scatter(x, y, c='blue', edgecolor='k', s=100, label=f'AP<{Xap}')

        # Compute correlation only for points in the region x>10
        x_X = x[x > X]
        y_X = y[x > X]

        # Ensure both are 1D arrays
        x_X = np.array(x_X).astype(float)
        y_X = np.array(y_X).astype(float)

            # Print the data for debugging
            # print(f"User ID: {user_id}")
            # print("x_gt_10:", x_gt_10)
            # print("y_gt_10:", y_gt_10)
            # print("===" * 10)

        if len(x_X)>1:
                corr = np.corrcoef(x_X, y_X)[0, 1]
        else:
                corr = "N/A"

        plt.title(f'Ess deviation vs pa deviation for {user_id}. Correlation (x>{X}): {corr}')
        plt.xlabel('ess_dev')
        plt.ylabel('pa_dev')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f'dev_correlation_{user_id}.png', dpi=300)
        plt.close()
#%%
# ------------------- ARTIFACT PERCENTAGE < 5 % -------------------

df['pa_ap_base_30'] = df.apply(
    lambda row: compute_baseline(row, df, 'pa', 30, 1, 5),
    axis=1
)

df['pa_ap_base_7'] = df.apply(
    lambda row: compute_baseline(row, df, 'pa', 7, 1, 5),
    axis=1
)

print(df.info())
#%%
# print(df[['pa_base_7', 'pa_base_30', 'pa_ap_base_7', 'pa_ap_base_30']].head())

print(len(df[~df['pa_ap_base_7'].isna() & df['pa_base_7'].isna()]))
print(len(df[~df['pa_ap_base_30'].isna() & df['pa_base_30'].isna()]))
#%%
# check the error we are making in baseline computations using all datapoints vs datapoints with AP < 5 where both available

mask = ~df['pa_ap_base_7'].isna() & (df['pa_ap_base_7'] != 0)
df['pa_err_artifact_7'] = np.where(mask, ((df['pa_base_7'] - df['pa_ap_base_7']) / df['pa_ap_base_7']) * 100, None)

mask = ~df['pa_ap_base_30'].isna() & (df['pa_ap_base_30'] != 0)
df['pa_err_artifact_30'] = np.where(mask, ((df['pa_base_30'] - df['pa_ap_base_30']) / df['pa_ap_base_30']) * 100, None)
#%%
# plot the errors made as a function of artifact percentage

for user, group in df.groupby('user_id'):

    # Filter rows where data available

    err7_data = group.dropna(subset=['pa_err_artifact_7'])[['artifact_percentage', 'pa_err_artifact_7']]
    err30_data = group.dropna(subset=['pa_err_artifact_30'])[['artifact_percentage', 'pa_err_artifact_30']]

    plt.scatter(err7_data['artifact_percentage'], err7_data['pa_err_artifact_7'], c='cornflowerblue', edgecolor='k', s=100, label='7-days')
    plt.scatter(err30_data['artifact_percentage'], err30_data['pa_err_artifact_30'], c='lightcoral', edgecolor='k', s=100, label='30-days')

    plt.title(f'Errors of 7 (and 30)-days baseline considering AP>5: {user}')
    # Adjust x-axis labels
    plt.xlabel('Artifact Percentage')
    plt.ylabel('% error')
    # plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'baseline_err_artifact_{user}.png')
    plt.close()
#%%
# re-plot considering AP<5 data only now

import matplotlib.pyplot as plt


# Group by user_id and plot data for each user
for user, group in df.groupby('user_id'):

    # Filter rows where pa and ess_tot are available

    ess_base30_data = group.dropna(subset=['ess_base_30'])[['date', 'ess_base_30']]
    ess_base7_data = group.dropna(subset=['ess_base_7'])[['date', 'ess_base_7']]


    pa_base30_data = group.dropna(subset=['pa_ap_base_30'])[['date', 'pa_ap_base_30']]
    pa_base7_data = group.dropna(subset=['pa_ap_base_7'])[['date', 'pa_ap_base_7']]

    pa_data = group.dropna(subset=['pa'])
    pa_data = pa_data[pa_data['artifact_percentage'] < 5][['date', 'pa']]
    ess_tot_data = group.dropna(subset=['ess_tot'])[['date', 'ess_tot']]

    # Plotting pa data
    plt.scatter(pa_data['date'], pa_data['pa'], marker='o', label='pa', color='lightblue')
    plt.plot(pa_base7_data['date'], pa_base7_data['pa_ap_base_7'], label='pa weekly baseline', color='cornflowerblue')
    plt.plot(pa_base30_data['date'], pa_base30_data['pa_ap_base_30'], label='pa monthly baseline', color='darkslateblue')

    # Plotting ess_tot data
    plt.scatter(ess_tot_data['date'], ess_tot_data['ess_tot'], marker='x', label='ess', color='orange')
    plt.plot(ess_base7_data['date'], ess_base7_data['ess_base_7'], label='ess weekly baseline', color='lightcoral')
    plt.plot(ess_base30_data['date'], ess_base30_data['ess_base_30'], label='ess monthly baseline', color='darkred')

    plt.title(f'pa (artifacts < 5 %) and ess for user: {user}')
    # Adjust x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.ylim(0, 400)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'pa_ess_{user}_artifact.png')
    plt.close()
#%%
print(df.info())
#%%
# define deviations (first workout of the day) and compute correlations

mask0 = ~df['date_diff'].isna() & (df['date_diff'] != 0)
mask = mask0 & ~df['pa_ap_base_7'].isna() & (df['pa_ap_base_7'] != 0) & ~df['pa_ap_base_30'].isna() & (df['pa_ap_base_30'] != 0)  # Not NaN and not zero  # Not NaN and not zero
df['pa_ap_dev'] = np.where(mask, ((df['pa_ap_base_7'] - df['pa_ap_base_30']) / df['pa_ap_base_30']) * 100, None)

# plot correlations

correlation_df = df[
    (~df['ess_dev'].isna())
    & (~df['pa_ap_dev'].isna())
    # & (df['ess_dev'] > 10)
    ]

# Threshold for correlation
X = 20

for user_id in correlation_df['user_id'].unique():
    user_data = correlation_df[correlation_df['user_id'] == user_id]

    print(len(user_data))

    x = user_data['ess_dev'].values
    y = user_data['pa_ap_dev'].values

    # Ensure have more than one entry to plot
    if len(x) > 1:
        plt.figure(figsize=(10, 6))

        plt.axvspan(X, max(x), color='yellow', zorder=0)  # zorder sets the stacking order
        # Add vertical dashed black line at x=10
        plt.axvline(x=X, color='black', linestyle='--')

        plt.scatter(x, y, c='blue', edgecolor='k', s=100, label=f'AP < 5 %')

        # Compute correlation only for points in the region x>X
        x_X = x[x > X]
        y_X = y[x > X]

        # Ensure both are 1D arrays
        x_X = np.array(x_X).astype(float)
        y_X = np.array(y_X).astype(float)

        # Print the data for debugging
        # print(f"User ID: {user_id}")
        # print("x_gt_10:", x_gt_10)
        # print("y_gt_10:", y_gt_10)
        # print("===" * 10)

        if len(x_X) > 1:
            corr = np.corrcoef(x_X, y_X)[0, 1]
        else:
            corr = "N/A"

        plt.title(f'Ess deviation vs pa (AP < 5 %) deviation for {user_id}. Correlation (x>{X}): {corr}')
        plt.xlabel('ess_dev')
        plt.ylabel('pa_dev')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f'dev_correlation_{user_id}_artifact.png', dpi=300)
        plt.close()
#%%
# study the trends of pa monthly baseline wrt monthly ess baseline,
# using AP < 5 %

# use only users with first workouts with AP < 5 mostly
# For each user_id, consider all rows with date_diff=float (to exclude None and NaN) and date_diff not zero. If at least 80% of these rows have row[artifact_percentage] < 5, save the user_id in the users_ap list

# Filter valid rows
valid_rows = df[(~df['date_diff'].isna()) & (df['date_diff'] != 0)]
# Group by user_id and compute the percentage of rows with artifact_percentage < 5
grouped = valid_rows.groupby('user_id').apply(lambda group: sum(group['artifact_percentage'] < 5) / len(group))
# Get user_ids that satisfy the condition
users_ap = grouped[grouped >= 0.8].index.tolist()

print(users_ap)
print(len(users_ap))

# Step 2
mask0 = ~df['date_diff'].isna() & (df['date_diff'] != 0)
mask = (
    mask0 &
    ~df['pa_ap_base_30'].isna() &
    ~df['ess_base_30'].isna() &
    (df['ess_base_30'] != 0) &
    df['user_id'].isin(users_ap)
)
df['trend_ap'] = np.where(mask, df['pa_ap_base_30'] / df['ess_base_30'], np.nan)
#%%
# plot it
filtered_df = df[df['user_id'].isin(users_ap)]

for user, group in filtered_df.groupby('user_id'):

    # Filter rows where data available
    trend_data = group.dropna(subset=['trend_ap'])[['date', 'trend_ap']]

    plt.plot(trend_data['date'], trend_data['trend_ap'], label='pa/ess', color='green')

    plt.title(f'pa/ess for user {user} (AP < 5 %)')
    # Adjust x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Date')
    plt.ylabel('pa/ess')
    # plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'trends_{user}.png')
    plt.close()
#%%
# 1. study of the day after huge effort wrt monthly ess baseline, pa has decreased wrt to YESTERDAY's value (define deviations using date_diff = 1)

# 2. study of the day after huge effort wrt monthly ess baseline, pa has decreased wrt to monthly value (define all workout deviations wrt to monthly baseline)

# SEPARATE FILE (focus on AP < 5%)

# 3. Re-plot (separate file) p vs alpha1 at different time windows (total vs monthly data) and define the slope and intercept

# 4. Study DURABILITY (first half/second half of exercise and check trend vs Training Peaks Power/HR drop)

# 5. Figure out whether pa can be defined using different functions of p and alpha1

# In few months: re-do this analysis considering
# - ess from running
# - cleaning all power, alpha1, hr data where artifact appear, to compute a cleaner pa