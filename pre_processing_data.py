'''
ADD SMOOTHENING LAG-DEP COLUMNS
'''
#%% 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_clean_3.csv')
# Drop specified columns
df = df.drop(['ess', 'lasttime', 'p0_timestamps', 'useless_p0_timestamps'], axis=1)
#%%
'''
- Restrict to data we are interested in, analyze it, and eventually create and save a df
- Use the interested df columns in other analysis (eg ESS or)
'''
# Filter rows where artifact_percentage is less than 5%
df = df[df['artifact_percentage'] < 5]
print(df.info())
#%%
# unique_types = df['tuples_raw'].apply(type).unique()
# print(unique_types)
#%%
# they are strings or float (nan), so convert the strings into list
# and keep rows with only lists
df['local_date'] = pd.to_datetime(df['local_date'])

import ast
def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s  # Return original value if conversion fails

df['tuples_raw'] = df['tuples_raw'].apply(string_to_list)
# Keep rows where tuples_raw is a list
df = df[df['tuples_raw'].apply(lambda x: isinstance(x, list))]
print(df.info())
print(df['tuples_raw'].head())
# %%
''' FUNCTIONS:
Split tuples into first interval (no fatigue) and remaining interval, etc ...
'''
# Select tuples spanning 20/25/30 minutes of activity:
# starting at time with p not zero 
# OR
# time_delta minutes later because of artifacts

import numpy as np

def split_tuples(tuples_list, time_delta, duration):
    if not isinstance(tuples_list, list) or not tuples_list:
        return (np.nan, np.nan)
    # sort to ensure order of timestamps
    tuples_list = sorted(tuples_list, key=lambda x: x[0])
    # Find the first timestamp with power not 0
    time_start = next((t[0] for t in tuples_list if t[1] != 0), None)
    if time_start is None:  # If all tuples have power 0
        return (np.nan, np.nan)
    # Adjust start time by time_delta if provided
    time_start += time_delta
    # Determine intervals
    time_middle = time_start + duration 
    beginning_tuples = [t for t in tuples_list if time_start <= t[0] and t[0] <= time_middle]
    # last interval: (duration) mins before end
    time_end = max(t[0] for t in tuples_list if t[1] != 0)
    time_start_2 = time_end - duration
    if not beginning_tuples or time_start_2 < beginning_tuples[-1][0]:
        remaining_tuples = np.nan
    else:
        remaining_tuples = [t for t in tuples_list if time_start_2 <= t[0] and t[0] <= time_end]
        if len(remaining_tuples) == 0:
            remaining_tuples = np.nan
    return (beginning_tuples, remaining_tuples)


def calculate_activity_ratio(tuples_list, active_mins):
    if not isinstance(tuples_list, list) or len(tuples_list) == 0:
        return np.nan
    time_diff_seconds = tuples_list[-1][0] - tuples_list[0][0]
    if time_diff_seconds <= 0:
        return np.nan
    ratio = (active_mins * 60) / time_diff_seconds
    return round(ratio, 2) 


# Calculate the percentages of ACTIVITY below threshold
def fraction_below_threshold(tuples_list, threshold):
    if not isinstance(tuples_list, list):
        return np.nan
    # consider only active timestamps 
    tuples_list = [t for t in tuples_list if t[1] != 0]
    if not tuples_list:
        return np.nan
    count_below_threshold = sum(1 for _, _, a1_entry, _ in tuples_list if a1_entry < threshold)
    ratio = count_below_threshold / len(tuples_list)
    return round(ratio, 3)  # Round the ratio to X decimal places after the decimal point


def smoothening_tuples(tuples_list, smoothening_interval):
    # Convert tuples_list to DataFrame
    data_df = pd.DataFrame(tuples_list, columns=['timestamp', 'power', 'alpha1', 'heartrate'])
    # Convert timestamp to datetime format and set it as index
    data_df['timestamp_dt'] = pd.to_datetime(data_df['timestamp'], unit='s')
    data_df.set_index('timestamp_dt', inplace=True)
    # Compute rolling average for power and alpha1
    data_df['rolling_power'] = data_df['power'].rolling(window=f'{smoothening_interval}s').mean().round(2)
    data_df['rolling_alpha1'] = data_df['alpha1'].rolling(window=f'{smoothening_interval}s').mean().round(2)
    # Create a new list of tuples (timestamp, rolling_power, rolling_alpha1, heartrate)
    new_tuples_list = [tuple(x) for x in data_df[['timestamp', 'rolling_power', 'rolling_alpha1', 'heartrate']].to_records(index=False)]
    return new_tuples_list

#%%
# Info on entire workouts
df['active_mins'] = df['tuples_raw'].apply(
    lambda tuples_list: np.nan if not isinstance(tuples_list, list) else sum(1 for t in tuples_list if t[1] != 0)/12
)
df['activity_ratio'] = df.apply(
    lambda row: calculate_activity_ratio(row['tuples_raw'], row['active_mins']),
    axis=1
)
df['dynamic_range'] = df['tuples_raw'].apply(lambda x: fraction_below_threshold(x, 1))
df['below_75'] = df['tuples_raw'].apply(lambda x: fraction_below_threshold(x, 0.75))
df['below_50'] = df['tuples_raw'].apply(lambda x: fraction_below_threshold(x, 0.50))
print(df.info())
#%%
# smoothening before splitting and deleting
smoothening_seconds = 60
df['smooth_tuples'] = df['tuples_raw'].apply(lambda x: smoothening_tuples(x, smoothening_seconds))
print(df[['tuples_raw', 'smooth_tuples']].head(5))
#%%
# split workouts in 2 parts:
duration_mins = 25
time_delta_mins = 5

df[['first_part', 'remaining_part']] = df.apply(
    lambda row: split_tuples(
        row['tuples_raw'], 
        time_delta_mins*60, 
        duration_mins*60,
    ), 
    axis=1, result_type='expand'
)
# drop the raw tuples
df.drop('tuples_raw', axis=1, inplace=True)

# same for smoothened data
df[['first_part_smooth', 'remaining_part_smooth']] = df.apply(
    lambda row: split_tuples(
        row['smooth_tuples'], 
        time_delta_mins*60, 
        duration_mins*60,
    ), 
    axis=1, result_type='expand'
)
# drop the raw tuples
df.drop('smooth_tuples', axis=1, inplace=True)

print(df[['first_part', 'remaining_part']].head(10))
print(df[['first_part_smooth', 'remaining_part_smooth']].head(10))
print(df.info())
#%%
df['active_mins_first'] = df['first_part'].apply(
    lambda tuples_list: np.nan if not isinstance(tuples_list, list) else sum(1 for t in tuples_list if t[1] != 0)/12
)
df['active_mins_remaining'] = df['remaining_part'].apply(
    lambda tuples_list: np.nan if not isinstance(tuples_list, list) else sum(1 for t in tuples_list if t[1] != 0)/12
)

df['activity_ratio_first'] = df.apply(
    lambda row: calculate_activity_ratio(row['first_part'], row['active_mins_first']),
    axis=1
)
df['activity_ratio_remaining'] = df.apply(
    lambda row: calculate_activity_ratio(row['remaining_part'], row['active_mins_remaining']),
    axis=1
)

print(df.info())
#%%
df['dynamic_range_first'] = df['first_part'].apply(lambda x: fraction_below_threshold(x, 1))
df['below_75_first'] = df['first_part'].apply(lambda x: fraction_below_threshold(x, 0.75))
df['below_50_first'] = df['first_part'].apply(lambda x: fraction_below_threshold(x, 0.50))
df['dynamic_range_remaining'] = df['remaining_part'].apply(lambda x: fraction_below_threshold(x, 1))
df['below_75_remaining'] = df['remaining_part'].apply(lambda x: fraction_below_threshold(x, 0.75))
df['below_50_remaining'] = df['remaining_part'].apply(lambda x: fraction_below_threshold(x, 0.50))
print(df.info())
# %%
df.to_csv('df_prep.csv', index=False)
# %%

