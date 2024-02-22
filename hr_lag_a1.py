#%%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_clean_3.csv')
# Drop specified columns
df = df.drop(['ess', 'lasttime', 'p0_timestamps', 'useless_p0_timestamps'], axis=1)
print(df.info())
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
#%%
df = df[df['tuples_raw'].apply(lambda x: isinstance(x, list))]
print(df.info())
#%%
'''
COMPUTE AVERAGES since needed in the future computation:
- restrict to workouts longer than 45 minutes and where at least 80% p not zero
- for workout that do not satisfy this constraint: averages = np.nan
'''
def compute_averages(tuples_raw):
    # Check if the workout is longer than 45 minutes
    if tuples_raw[-1][0] < 45 * 60:  # Assuming the first element of each tuple is the timestamp in seconds
        return np.nan, np.nan
    workout_df = pd.DataFrame(tuples_raw, columns=['timestamp', 'power', 'alpha1', 'hr'])
    # Check if at least 80% of power readings are non-zero
    if (workout_df['power'] > 0).mean() < 0.8:
        return np.nan, np.nan
    power_avg = workout_df['power'].mean()
    alpha1_avg = workout_df['alpha1'].mean()
    return power_avg, alpha1_avg

# Apply the function to each row
df['power_avg'], df['alpha1_avg'] = zip(*df['tuples_raw'].apply(compute_averages))
# %%
# Filter rows where artifact_percentage is less than 5% and averages are available
df = df[(df['artifact_percentage'] < 5) & (~pd.isna(df['power_avg'])) & (~pd.isna(df['alpha1_avg']))]
print(df.info())
# %%
print(df['tuples_raw'].head())
# %%
# Select tuples spanning 20/25/30 minutes of activity:
# starting at time with p not zero 
# OR
# time_delta minutes later because of artifacts

import numpy as np

def correct_tuples(tuples_list):
    if not isinstance(tuples_list, list):
        return tuples_list

    missing_tuples = []
    for current_tuple, next_tuple in zip(tuples_list, tuples_list[1:]):
        # Calculate the time difference between consecutive tuples
        time_diff = math.floor(next_tuple[0]) - math.floor(current_tuple[0])

        if time_diff == 5 or time_diff % 5 != 0:
            if time_diff % 5 != 0:
                print(f'WARNING: time difference: {time_diff}')
                return np.nan
            continue

        num_missing_tuples = time_diff // 5 - 1  # Calculate how many tuples are missing
        # Add missing tuples with NaN values for other entries
        for j in range(int(num_missing_tuples)):
                missing_timestamp = current_tuple[0] + (j + 1) * 5
                missing_tuples.append((missing_timestamp, np.nan, np.nan, np.nan))

    # Append missing tuples to the original list and then sort the entire list
    tuples_list.extend(missing_tuples)
    tuples_list.sort(key=lambda x: x[0])

    return tuples_list

def select_tuples(tuples_list, 
                  time_delta=None, 
                  duration=None, 
                  dynamic_a1_pct=None, 
                  excluding_p0=False):
    if not isinstance(tuples_list, list) or not tuples_list:
        return np.nan
    # sort to ensure order of timestamps
    tuples_list = sorted(tuples_list, key=lambda x: x[0])

    # Find the first timestamp with power not 0
    time_start = next((t[0] for t in tuples_list if t[1] != 0), None)
    if time_start is None:  # If all tuples have power 0
        return np.nan

    # Adjust start time by time_delta if provided
    if time_delta is not None:
        time_start += time_delta

    # Set end time if duration is provided
    time_end = time_start + duration if duration is not None else None

    # Filter tuples based on the time window and power condition
    selected_tuples = [t for t in tuples_list if time_start <= t[0] and (time_end is None or t[0] <= time_end) 
                       and (excluding_p0 is False or t[1] != 0)]
    # correct them
    selected_tuples = correct_tuples(selected_tuples)
    
    # accept only those with a minimum a1 in the dynamic range
    if dynamic_a1_pct:
        # Filter tuples where second entry is a finite number
        finite_tuples = [t for t in selected_tuples if np.isfinite(t[1])]
        if len(finite_tuples)==0 or sum(t[2] > 1 for t in finite_tuples) / len(finite_tuples) > 1 - dynamic_a1_pct:
            return np.nan
        
    return selected_tuples
#%%
'''
# check whether it works as a projetion (as it should): YES !!!
import pandas as pd
df['tuples_new'] = df.apply(lambda row: select_tuples(row['tuples_raw'], time_delta=5*60, duration=20*60), axis=1)
# Step 2: Check if correct_tuples returns the same list
df['tuples_new_2'] = df['tuples_new'].apply(correct_tuples)
# remember that nan == nan is FALSE by default and comparison must be done within lists
df['function_is_projective'] = df.apply(
    lambda row: (not isinstance(row['tuples_new'], list) and not isinstance(row['tuples_new_2'], list)) 
                or (isinstance(row['tuples_new'], list) and isinstance(row['tuples_new_2'], list) and row['tuples_new'] == row['tuples_new_2']),
    axis=1
)
count_not_projective = df['function_is_projective'].value_counts().get(False, 0)
print(f"Number of rows where function is not projective: {count_not_projective}")
print(df[df['function_is_projective'] == False][['tuples_raw', 'tuples_new', 'tuples_new_2', 'function_is_projective']].head(20))
# Step 3: Count the number of rows where tuples_raw is a list but tuples_new is not
count_mismatch = sum(df.apply(lambda row: isinstance(row['tuples_raw'], list) and not isinstance(row['tuples_new'], list), axis=1))
print(f"Number of rows where 'tuples_raw' is a list but 'tuples_new' is not: {count_mismatch}")
# Step 4: Delete the tuples_new column
df.drop(['tuples_new', 'tuples_new_2', 'function_is_projective'], axis=1, inplace=True)
'''
#%%
'''
Define a new tuple column, selecting 20 mins duration 
and restricting to those with 50% data a1<1
'''
df['selected_data'] = df.apply(lambda row: select_tuples(
    row['tuples_raw'], 
    time_delta=5*60, 
    duration=25*60,
    dynamic_a1_pct=0.5), 
    axis=1)
# df_data = df[df['selected_data'].apply(lambda x: isinstance(x, list))]
#%%
print(df.info())
# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
n_rows = 2
rolling_window = 2 # each one is 5 seconds, so 3 = 15 seconds

for user_id in df['user_id'].unique():
    # Filter the DataFrame for the current user ID 
    user_df = df[(df['user_id'] == user_id) & (df['selected_data'].apply(isinstance, args=(list,)))]
    print(f"User {user_id} has {len(user_df)} interesting workouts")
    if len(user_df) == 0:
        print("Moving to the next user...")
        continue
    
    user_df = user_df.reset_index(drop=True)
    user_df = user_df.head(n_rows)

    for row_index, row in user_df.iterrows():
        # Convert row data to a DataFrame for ease 
        data_df = pd.DataFrame(row['selected_data'], 
                               columns=['timestamp', 'power', 'alpha1', 'heartrate'])
        
        # moving average for power
        data_df['rolling_power'] = data_df['power'].rolling(
            window=12, 
            min_periods=1).mean()

        # smoothening of alpha1 using 2 values before and after
        data_df['smooth_a1'] = data_df['alpha1'].rolling(
            window=5, 
            min_periods=1, 
            center=True).mean()

        # Create the figure and the first axis
        fig, ax1 = plt.subplots(figsize=(12, 10))

        # Plot power
        color = 'tab:blue'
        ax1.set_xlabel('timestamp')
        ax1.set_ylabel('power', color=color)
        ax1.scatter(data_df['timestamp'], data_df['power'], color='lightblue')
        ax1.plot(data_df['timestamp'], data_df['rolling_power'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create another y-axis for alpha1
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('alpha1', color=color)
        ax2.scatter(data_df['timestamp'], data_df['alpha1'], color='yellow')
        ax2.plot(data_df['timestamp'], data_df['alpha1'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Create another y-axis for heartrate
        ax3 = ax1.twinx()  
        ax3.spines['right'].set_position(('outward', 40))
        color = 'tab:green'
        ax3.set_ylabel('heartrate', color=color)
        # ax2.scatter(data_df['timestamp'], data_df['alpha1'], color='yellow')
        ax3.plot(data_df['timestamp'], data_df['heartrate'], color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        # Making sure the x-axis labels (timestamps) are not too crowded
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))

        # Save the plot
        plt.title(f'User ID: {user_id}')
        plt.savefig(f'heart_lags/p_a1_lag/{user_id}_{row_index}.png')
        plt.close(fig)
# %%
'''
OBSERVATIONS:

1. alpha1 is impressively smooth, meaning that we better find a way to clean 
   raw data to exclude WILD fluctuations (maybe > 3 sigma) ----> Do in another file

2. HERE: use the idea that the power should be integrated over the lag interval
         and check how the p-a1 correlation IN THE DYNAMIC RANGE changes with it

ATTEMPTS:
- p lagged vs a1 (lagged or shifted): inconclusive 
(other file)
- p lagged and shifted vs a1 (and variations): inconclusive 
(attempt 1 here)
- shifting pointless if we are already using lagged power

IMPORTANT:
- DFA-a1 moves much slower than power (bigger inertia), 
so shifting simply does not work
- it makes sense to use lagged power (integrating over a past interval)
such that the power moves much slower and may match alpha1
- remember: integrating over past 1 min -> introducing 1 min lag
- Attempt with variations wrt effective averages
'''

# ATTEMPT 1:

import scipy.stats as stats

def identify_power_regions(data_df):
    regions = []
    
    window_size = 6
    # Calculate the rolling average of power_var 
    # over a window of 6 rows and shift backward
    rolling_avg = data_df['power_var'].rolling(
        window=window_size).mean().shift(-(window_size-1))

    i = 0
    while i < len(data_df) - window_size:
        if (data_df['power_var'].iloc[i] > 0 
            and data_df['power_var'].iloc[i+1] > 0 
            and data_df['power_var'].iloc[i:i+2].mean() > 0.05):
            r1 = i

            # Search for r2 using the shifted rolling average
            for j in range(r1 + 1, len(data_df) - window_size + 1):
                if np.isnan(data_df['power_var'].iloc[j]) or rolling_avg.iloc[j] < -0.05:
                    r2 = j
                    if r2 - r1 > 6:
                        regions.append((r1, r2))
                        i = r2
                        break
            else:
                i += 1
                continue
        i += 1

    return regions if regions else np.nan


n_rows = 5
for user_id in df['user_id'].unique():
    # Filter the DataFrame for the current user ID and select the first n_rows
    user_df = df[(df['user_id'] == user_id) & (df['selected_data'].apply(isinstance, args=(list,)))]
    print(f"User {user_id} has {len(user_df)} interesting workouts")
    if len(user_df) == 0:
        print("Moving to the next user...")
        continue
    
    user_df = user_df.reset_index(drop=True)
    user_df = user_df.head(n_rows)

    for row_index, row in user_df.iterrows():
        # Convert row data to a DataFrame for ease 
        data_df = pd.DataFrame(row['selected_data'], 
                               columns=['timestamp', 'power', 'alpha1', 'heartrate'])
        
        
        power_avg = data_df['power'].mean()
        # defining relative variations with respect to previous entry
        # using avg_power/2 as minimum threshold
        data_df['power_var'] = np.where(
            (data_df['power'] > power_avg/2) & (data_df['power'].shift(1) > power_avg/2),
            data_df['power'].diff() / data_df['power'].shift(1),
            np.nan
        )
        # variations for a1 have a minimum threshold a1=0.65 to avoid high intensity domain 
        # where fatigue plays an important role
        data_df['a1_var'] = np.where(
            (data_df['alpha1'] > 0.5) & (data_df['alpha1'].shift(1) != 0),
            data_df['alpha1'].diff() / data_df['alpha1'].shift(1),
            np.nan
        )

        # smoothen out both power and a1 variations using 3 previous values:
        # In this way, requiring the smoothened values to satisfy some condition is tantamount to
        # ensuring that condition is met in the last 3 entries
        # Smooth out both 'power_var' and 'a1_var' using a rolling window of 3
        data_df['power_var'] = data_df['power_var'].rolling(window=3, min_periods=3).mean()
        data_df['a1_var'] = data_df['a1_var'].rolling(window=3, min_periods=3).mean()

        # Focus on positive increase in power
        data_df['power_var'] = np.where(
            data_df['power_var']>0,
            data_df['power_var'],
            np.nan
        )

        # Normalize variations by dividing by its average value
        # Calculate the mean of 'power_var' excluding NaN values
        power_var_avg = abs(data_df['power_var']).mean()
        a1_var_avg = abs(data_df['a1_var']).mean()
        # Perform the normalization
        data_df['power_var'] = data_df['power_var'] / power_var_avg
        data_df['a1_var'] = data_df['a1_var'] / a1_var_avg

        # Filter the normalized power variations:
        data_df['power_var'] = np.where(
            data_df['power_var'] > 1,
            data_df['power_var'],
            np.nan
        )
        



        # separate positive / negative variations
        # for head-lag
        data_df['power_var_p'] = np.where(
            data_df['power_var']>0,
            data_df['power_var'],
            np.nan
        )
        data_df['a1_var_p'] = np.where(
            data_df['a1_var']>0,
            data_df['a1_var'],
            np.nan
        )

        # for tail-lag
        data_df['power_var_n'] = np.where(
            data_df['power_var']<0,
            data_df['power_var'],
            np.nan
        )
        data_df['a1_var_n'] = np.where(
            data_df['a1_var']<0,
            data_df['a1_var'],
            np.nan
        )


        # power_regions = identify_power_regions(data_df)
        # print(power_regions)

        # Create the figure and the first axis
        fig, ax1 = plt.subplots(figsize=(12, 10))
        
        color = 'tab:blue'
        ax1.set_xlabel('timestamp')
        ax1.set_ylabel('consecutive difference power', color=color)
        ax1.scatter(data_df['timestamp'], data_df['power_var'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create another y-axis for alpha1
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('consecutive difference a1', color=color)
        ax2.scatter(data_df['timestamp'], data_df['a1_var'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        

        '''
        # Create another y-axis for power 
        ax3 = ax1.twinx()  
        # Change the edge color and line width of the right spine 
        # of ax2 (to hide it)
        ax3.spines['right'].set_color('none')
        # Move the new y-axis to the left side
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        ax3.spines['left'].set_position(('outward', 40))  
        ax3.spines['left'].set_visible(True)
        color = 'tab:blue'
        ax3.set_ylabel('power', color=color)
        ax3.plot(data_df['timestamp'], data_df['power'], color='lightblue', linestyle='--', linewidth=4)
        ax3.tick_params(axis='y', labelcolor=color)


        # Create another y-axis for a1 
        ax4 = ax1.twinx()  
        ax4.spines['right'].set_position(('outward', 40))
        color = 'tab:red'
        ax4.set_ylabel('alpha1', color=color)
        ax4.plot(data_df['timestamp'], data_df['alpha1'], color='orange', linestyle='--', linewidth=4)
        ax4.tick_params(axis='y', labelcolor=color)
        '''

        
        '''
        if isinstance(power_regions, list):
            for start, end in power_regions:
                # if end - start > 5:
                    # Convert start and end indices to timestamps
                    start_timestamp = data_df.iloc[start]['timestamp']
                    end_timestamp = data_df.iloc[end]['timestamp']
                    print(f'{[(start, end)]} means {[start_timestamp, end_timestamp]}')
                    # Add a vertical span (band) to the plot
                    ax1.axvspan(start_timestamp, end_timestamp, color='yellow', alpha=0.5)
        '''
        # Making sure the x-axis labels (timestamps) are not too crowded
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # Save the plot
        plt.title(f'User ID: {user_id}')
        plt.savefig(f'heart_lags/p_a1_lag/{user_id}_{row_index}.png')
        plt.close(fig)

        correlations = []
        lags = range(0, 10)  # 0 to 45 seconds in steps of 5 seconds
        # also plot correlations for 0 - 40 seconds
        for lag in lags:           
            # Shift 'power_var' by 'lag': automatically firstlag values are filled with nans
            data_df['power_var_shifted'] = data_df['power_var'].shift(lag)
            # Compute Spearman correlation if there are at least few meaningful pairs
            if data_df[['power_var_shifted', 'a1_var']].dropna().shape[0] >= 5:
                corr, _ = stats.spearmanr(data_df['power_var_shifted'], data_df['a1_var'], nan_policy='omit')
                correlations.append(corr)
            else:
                correlations.append(np.nan)

            '''
            data_df['power_lagged'] = data_df['power']
            if lag > 0:
                data_df['power_lagged'] = data_df['power'].rolling(
                                                            window=lag, min_periods=1).mean()
            data_df['power_var_lagged'] = np.where(
                    (data_df['power_lagged'] > power_avg/2) & (data_df['power_lagged'].shift(1) > power_avg/2),
                    data_df['power_lagged'].diff() / data_df['power_lagged'].shift(1),
                    np.nan
                    )
            data_df['power_var_lagged_p'] = np.where(
                    data_df['power_var_lagged']>0,
                    data_df['power_var_lagged'],
                    np.nan
                    )
            # Compute Spearman correlation if there are at least 20 meaningful pairs
            if data_df[['power_var_lagged', 'a1_var']].dropna().shape[0] >= 10:
                corr, _ = stats.spearmanr(data_df['power_var_lagged'], data_df['a1_var'], nan_policy='omit')
                correlations.append(corr)
            else:
                correlations.append(np.nan)
            '''

        # Plot correlations as a function of lag
        plt.figure(figsize=(10, 6))
        plt.plot([lag * 5 for lag in lags], correlations, marker='o')  # Multiplying lag by 5 for seconds
        plt.xlabel('Lag (seconds)')
        plt.ylabel('Spearman Correlation')
        plt.title(f'Correlations for Variations Power vs a1 - User ID: {user_id}')
        plt.grid(True)
        # Save the plot
        plt.savefig(f'heart_lags/p_a1_lag/{user_id}_{row_index}_correlations.png')
        plt.close()


# CONCLUSION: Figuring out the lag is pointless. Other method!
# %%
'''
ATTEMPT 2: CONSIDERING ONLY BIG VARIATIONS WRT AVERAGE POWER AND A1
- Smoothen out a1 symmetrically using 3 points
- Introduce lagged power by integrating over [0 - 90s] interval
- Scatterplot relative variations p_lagged vs a1 
(wrt their eff averages)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

n_rows_toplot = 4
lag_values = range(0, 36)
# for stat analysis on lags and correlations
correlations_df = pd.DataFrame(columns=['user_id', 'local_date', 'lags_correlations'])

for user_id in df['user_id'].unique():
    # Filter the DataFrame for the current user ID and select the first n_rows
    user_df = df[(df['user_id'] == user_id) & (df['selected_data'].apply(isinstance, args=(list,)))]
    print(f"User {user_id} has {len(user_df)} interesting workouts")
    if len(user_df) == 0:
        print("Moving to the next user...")
        continue
    
    # user_df = user_df.reset_index(drop=True)
    # user_df = user_df.head(n_rows_toplot)

    toplot_counter = 0
    for row_index, row in user_df.iterrows():
        toplot_counter = toplot_counter + 1
        workout_date = row['local_date']
        
        # Convert row data to a DataFrame for ease 
        data_df = pd.DataFrame(row['selected_data'], 
                               columns=['timestamp', 
                               'power', 
                               'alpha1', 
                               'heartrate'])
        # choose the averages
        # p_avg = row['power_avg']
        # a1_avg = row['alpha1_avg']
        # wrt the selected time frame to analyze
        filtered_data_df = data_df[data_df['power'] != 0]
        p_avg = filtered_data_df['power'].mean()
        a1_avg = filtered_data_df['alpha1'].mean()
        del filtered_data_df

        if p_avg * a1_avg == 0:
            print('Averages are zero: skip to next workout')
            continue
        # smoothen alpha1 and compute relative variation
        # smoothen using 2 values before and after
        data_df['a1'] = data_df['alpha1'].rolling(
            window=3, 
            min_periods=1, 
            center=True).mean()
        data_df['a1_var'] = np.where(
            (data_df['a1'] > 0),
            data_df['a1'] / a1_avg - 1,
            np.nan
        )
        # for plotting reasons
        data_df['p'] = data_df['power'].rolling(
            window=3, 
            min_periods=1, 
            center=True).mean()
        
        # HERE: 
        # 1. loop for each lag value:
        #   - compute the rolling power and variation
        #   - compute the Spearman correlation among p_var and a1_var using only pairs 
        #       where both abs(p_var) and abs(a1_var) > 0.1
        # 2. at the end of the loop we have a dictionary with 
        #    - key = lag  
        #    - elements = correlation, rolling_power, p_var relative to that key
        # 3. plot each correlation as a function of the lag
        # 4. identify the 3 smallest correlations (closer to -1) and create 2 figures:
        #   - fig 1: 3 scatterplots of p_var vs a1_var using p_var 
        #           corresponding to 3 smallest correlations. 
        #           Color code: red for the smallest correlation, blue for the second,
        #           green for the third. Report the correlation value in the legend
        #   - fig 2: scatterplot data_df['power'] (lightblue) and data_df['a1'] (orange) 
        #           both vs data_df['timestamp']. Also, plot the 3 rolling_powers vs timestamp
        #           corresponding to the 3 smallest correlations. Use the same color code used in fig 1 
        #           and report the correlations values in the legend
        #           Notice that all power data must be normalized using power_avg


        # Dictionary to store rolling power and p_var for each lag
        rolling_data = {}
        correlations = {}
        for lag in lag_values:  # Replace lag_values with your list of lag values
            if lag == 0:
                data_df['rolling_power'] = data_df['power']
            else:
                data_df['rolling_power'] = data_df['power'].rolling(window=lag, min_periods=1).mean()
            data_df['p_var'] = np.where((data_df['rolling_power'] > p_avg/2) & 
                                        (data_df['rolling_power'] < 6*p_avg),
                                        data_df['rolling_power'] / p_avg - 1, np.nan)
            # Apply the constraint for abs(p_var) and abs(a1_var) > 0.1
            valid_data = data_df[(data_df['p_var'].abs() > 0.2) & (data_df['a1_var'].abs() > 0.1)]
            if len(valid_data) < 12 * 5: # at least 5 minutes worth of change of pace
                correlation = np.nan
            else:
                correlation = valid_data[['p_var', 'a1_var']].corr(method='spearman').iloc[0, 1]
            correlations[lag] = correlation
            # Store rolling power and p_var for each lag
            rolling_data[lag] = data_df[['timestamp', 'rolling_power', 'p_var', 'a1_var']]
        
        # Save info on lag-correlation
        if correlations:  # Check if correlations dictionary is not empty
            # Fill correlations_df
            new_row = pd.DataFrame({
                'user_id': [user_id],
                'local_date': [workout_date],
                'lags_correlations': [correlations]
            })
            correlations_df = pd.concat([correlations_df, new_row], ignore_index=True)

        if toplot_counter <= n_rows_toplot:
            # Plot correlations as a function of lag
            correlation_values = [correlations[lag] for lag in lag_values]
            adjusted_lags = [lag * 5 for lag in lag_values]  # Adjust lag values to seconds
            plt.plot(adjusted_lags, correlation_values, marker='o')
            plt.xlabel('Lag (seconds)')
            plt.ylabel('Spearman Correlation')
            plt.title(f'Correlations for Variations Power vs a1 - User ID: {user_id}')
            plt.grid(True)
            # Save the plot
            plt.savefig(f'heart_lags/p_a1_lag/{user_id}_{row_index}_correlations.png')
            plt.close()
            
            # Find the three smallest correlations
            smallest_corrs = sorted(correlations, key=correlations.get)[:3]
            print(smallest_corrs)

            # Plotting
            # Figure 1: Scatter plots for the three smallest correlations
            plt.figure(figsize=(12, 8))
            colors = ['brown', 'red', 'green']  # Colors for the three smallest correlations
            for i, lag in enumerate(smallest_corrs):
                lag_data = rolling_data[lag]
                lag_time = 5 * lag
                plt.scatter(lag_data['p_var'], lag_data['a1_var'], color=colors[i], alpha=1, label=f'Lag {lag_time}s (Correlation: {correlations[lag]:.2f})')
            plt.axvline(x=0.2, color='grey', linestyle='--', linewidth=2)
            plt.axvline(x=-0.2, color='grey', linestyle='--', linewidth=2)
            plt.axhline(y=0.1, color='grey', linestyle='--', linewidth=2)
            plt.axhline(y=-0.1, color='grey', linestyle='--', linewidth=2)
            plt.title('Scatter plots of p_var vs a1_var')
            plt.xlabel('p_var')
            plt.ylabel('a1_var')
            plt.legend()
            plt.grid(True)
            # Save the plot
            plt.savefig(f'heart_lags/p_a1_lag/{user_id}_{row_index}_correlations_2.png')
            plt.close()


            # Figure 2: Time series plot
            plt.figure(figsize=(12, 8))
            plt.scatter(data_df['timestamp'], data_df['power'] / row['power_avg'], color='lightblue', label='Normalized Power')
            plt.scatter(data_df['timestamp'], data_df['a1'], color='orange', label='Alpha1')
            plt.plot(data_df['timestamp'], data_df['p'] / row['power_avg'], color='blue', label='Normalized Power (smooth)')
            for i, lag in enumerate(smallest_corrs):
                lag_data = rolling_data[lag]
                lag_time = 5 * lag
                plt.plot(lag_data['timestamp'], lag_data['rolling_power'] / row['power_avg'], color=colors[i], label=f'Lag {lag_time}s (Correlation: {correlations[lag]:.2f})')
            plt.xlabel('Timestamp')
            plt.ylabel('Normalized Power')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'heart_lags/p_a1_lag/{user_id}_{row_index}_correlations_3.png')
            plt.close()

        del data_df
        
    # Invoke garbage collection
    gc.collect()

# %%
print(correlations_df.info())
print(correlations_df.head())
# %%
# Remove rows where all values in 'lags_correlations' are NaN
correlations_df = correlations_df[correlations_df['lags_correlations'].apply(
    lambda d: not all(pd.isna(v) for v in d.values())
)]
print(correlations_df.info())
print(correlations_df.head())
#%%
correlations_df.to_csv('correlations.csv', index=False)
#%%
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import gc
# Function to convert a string representation of a dictionary back to a dictionary
def convert_to_dict(x):
    try:
        # Replace 'nan' with a placeholder unique string
        x_cleaned = x.replace('nan', '"_NaN_"')
        # Safely evaluate the string as a Python expression
        dict_obj = ast.literal_eval(x_cleaned)
        # Replace the placeholder string with np.nan
        return {k: (np.nan if v == '_NaN_' else v) for k, v in dict_obj.items()}
    except ValueError as e:
        print(f"Error converting to dict: {e}. Value: {x}")
        return None

correlations_df = pd.read_csv('correlations.csv')
correlations_df['local_date'] = pd.to_datetime(correlations_df['local_date'])
correlations_df['lags_correlations'] = correlations_df['lags_correlations'].apply(convert_to_dict)
all_are_dicts = all(isinstance(item, dict) for item in correlations_df['lags_correlations'])
print("All entries in 'lags_correlations' are dictionaries:", all_are_dicts)
#%%
print(correlations_df.info())
print(correlations_df.head())
# %%
'''
Check whether higher lags have a tendency to provide a bigger amount of valid points:
Q: are there more correlations for higher lags than for smaller lags? 
A: There is a sweet spot indeed, for longer moving averages 
the power is just more and more constant
'''
#%%
'''
It may work: consider only significant variations form the average for power
1. Simplify the correlation values by defect, 
and take the smallest lag with the strongest correlation
2. Evaluate if lag can really be > 1 minute

Remember that this is NOT the time needed for alpha1 to STABILIZE at a given power (fixed output):
Ueful nonetheless in order to compute the representatives (by clustering) using the proper moving power
instead of rolling over alpha1 too (which introduces another lag in alpha1 - which sounds bad)
'''
import seaborn as sns

# Function to add annotations to the plots
def annotate_stats(ax, series):
    mean = series.mean()
    std = series.std()
    min_val = series.min()
    max_val = series.max()

    stats_text = f'Mean: {mean:.2f}\nStd: {std:.2f}\nMin: {min_val}\nMax: {max_val}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right')

# OVERALL ANALYSIS: NOT THE BEST - better proceed user by user
'''
# Descriptive statistics:
plt.figure()
counts, bins, patches = plt.hist(best_correlations_df['best_correlation'], bins=20, alpha=0.6, color='g', weights=np.ones(len(best_correlations_df['best_correlation'])) / len(best_correlations_df['best_correlation']))
plt.title('Relative Frequency of Best Correlation Values')
plt.xlabel('Best Correlation')
plt.ylabel('Relative Frequency (%)')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentages
annotate_stats(plt.gca(), best_correlations_df['best_correlation'])
plt.grid(True)
plt.savefig(f'heart_lags/p_a1_lag/Best_correlations_PDF_corr.png')
plt.close()

# Plot PDF for lag values
plt.figure()
counts, bins, patches = plt.hist(best_correlations_df['lag'], bins=20, alpha=0.6, color='b', weights=np.ones(len(best_correlations_df['lag'])) / len(best_correlations_df['lag']))
plt.title('Relative Frequency of Lag Values')
plt.xlabel('Lag')
plt.ylabel('Relative Frequency (%)')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format y-axis as percentages
annotate_stats(plt.gca(), best_correlations_df['lag'])
plt.grid(True)
plt.savefig(f'heart_lags/p_a1_lag/Best_correlations_PDF_lags.png')
plt.close()

sns.scatterplot(data=best_correlations_df, x='lag', y='best_correlation')
plt.title('Scatterplot of Lag vs Best Correlation')
plt.xlabel('Lag')
plt.ylabel('Best Correlation')
plt.savefig(f'heart_lags/p_a1_lag/Best_correlations_lags.png')
plt.close()
'''
# %%
'''
FOR EACH USER:

GOAL: identify the lag universally
HOW: For each user with at least 10 workouts, for each lag, study ALL the correlation distribution and check average

OBS: must collect ALL the lag-correlation info
OBS: AVERAGE IS NOT INFORMATIVE enough: use the % 
'''
#%%
import my_functions
import importlib
importlib.reload(my_functions)


# check here the statistics on correlation values
for user_id in correlations_df['user_id'].unique():
    user_df = correlations_df[correlations_df['user_id'] == user_id]
    if len(user_df) < 5:
        continue

    # Collect all non-NaN correlation values for this user
    all_correlations = []
    for lag_corr_dict in user_df['lags_correlations']:
        all_correlations.extend([corr for corr in lag_corr_dict.values() if (corr > -1.1 and corr < 1.1)])

    my_functions.plot_percentage_histogram(all_correlations, -1, 1, 0.1, 'Correlations', f'heart_lags/p_a1_lag/{user_id}_corrs_histo.png')
#%%
# for each user: take the first lag with correlation < - 0.6
corr_threshold = -0.6
lag_values = range(0, 24)

for user_id in correlations_df['user_id'].unique():
    user_df = correlations_df[correlations_df['user_id'] == user_id]
    
    total_workouts = len(user_df)
    if total_workouts < 10:
        continue

    # Dictionary to store the correlations and compute a mean
    lag_correlations_tot = {}
    # collect info on lags with strong correlations
    all_lags_below_threshold = []
    first_lags = [] # below threshold

    # Loop through each row and aggregate correlations for each lag
    for index, row in user_df.iterrows():
        # Variables to track the first occurrence of lags in this row
        first_lag = None

        for lag, corr in row['lags_correlations'].items():
            # Add correlation value to the lag_correlations dictionary
            if not np.isnan(corr):
                if lag not in lag_correlations_tot:
                    lag_correlations_tot[lag] = []
                lag_correlations_tot[lag].append(corr)

            # Check and update first occurrence of lags in this row
            if first_lag is None and corr < corr_threshold:
                first_lag = lag
            
            # Add lag below threshold
            if corr < corr_threshold:
                all_lags_below_threshold.append(lag)
        
        # Append the first lags for this row to the respective lists
        if first_lag is not None:
            first_lags.append(first_lag)

    # sort the dictionary using key values:
    lag_correlations_tot = {k: lag_correlations_tot[k] for k in sorted(lag_correlations_tot)}

    # Compute the mean correlation for each lag
    mean_correlations = {lag: sum(corrs) / len(corrs) 
                         for lag, corrs in lag_correlations_tot.items() if corrs}
    correlations_average_wrt_workouts = {lag: sum(corrs) / total_workouts 
                                         for lag, corrs in lag_correlations_tot.items() if corrs}

    # print(mean_correlations) # DEBUGGING
    
    # Plot mean correlations vs. lags
    # Separate keys and values for plotting
    lags = list(mean_correlations.keys())
    mean_corr_values = list(mean_correlations.values())
    mean_corr_values_wrt_workouts = list(correlations_average_wrt_workouts.values())
    plt.figure(figsize=(12, 8))
    plt.plot(lags, mean_corr_values, marker='o', linestyle='--', 
             color='red', label='Mean Correlation')
    plt.plot(lags, mean_corr_values_wrt_workouts, marker='o', linestyle='--', 
             color='blue', label='Mean Correlation wrt Workouts')
    plt.title(f'Mean Correlation vs. Lag - User ID: {user_id} - {total_workouts} workouts')
    plt.xlabel('Lag')
    plt.ylabel('Mean Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'heart_lags/p_a1_lag/{user_id}_lags_mean_corrs.png')
    plt.close()


    # consider smallest lag for each workout
    fig, ax = plt.subplots(figsize=(12, 10))
    bin_size = 1
    num_bins = int((max(lag_values) - min(lag_values) + 1) / bin_size)
    # Creating the bins array
    bins = [min(lag_values) + i * bin_size for i in range(num_bins + 1)]
    # Plotting the histogram
    ax.hist(first_lags, bins=bins, range=(min(lag_values), max(lag_values) + bin_size), 
            rwidth=0.9, weights=[100. / total_workouts] * len(first_lags))
    ax.set_xticks(bins)  # Set x-ticks to bin values
    # Formatting the histogram
    ax.set_xticks(bins)  # Set x-ticks to bin values
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency (%) of total workouts')
    ax.set_title(f'Histogram of first lags with correlation below {corr_threshold}')
    plt.savefig(f'heart_lags/p_a1_lag/{user_id}_first_lags.png')
    plt.close()

    # consider all lags for each workout and check % of total workouts below corr_threshold
    fig, ax = plt.subplots(figsize=(12, 10))
    # Plotting the histogram
    ax.hist(all_lags_below_threshold, bins=bins, range=(min(lag_values), max(lag_values) + bin_size), 
            rwidth=0.9, weights=[100. / total_workouts] * len(all_lags_below_threshold))
    ax.set_xticks(bins)  # Set x-ticks to bin values
    # Formatting the histogram
    ax.set_xticks(bins)  # Set x-ticks to bin values
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency (%) of total workouts')
    ax.set_title(f'Histogram of all lags with correlation below {corr_threshold} - {total_workouts} workouts')
    plt.savefig(f'heart_lags/p_a1_lag/{user_id}_all_lags.png')
    plt.close()

# %%
'''
CONCLUSIONS:
- LAG DEPENDS ON THE WORKOUT
- 1 MINUTE SEEMS TO BE ACCEPTABLE TO BE TAKEN 
'''
# %%
