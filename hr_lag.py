''' 
This file tries to identify the lag between the power output and the hr time series
by looking for the delay that yields the highest correlation power - hr 

NOTES:
1. Need to consider data with small artifact percentage since this affects the BPMs
2. Correlations are studied for the time series and fluctuations
3. Power output needs to be slightly smoothened out, plot it vs HR to see 

RESULTS:

CONCLUSIONS:

'''
#%%
import pandas as pd
import json
import numpy as np

desired_keys = ["user_id", 
                "activity_date", 
                "artifact_percentage",
                #"activity_altistream",
                #"activity_cadencestream",
                "activity_hrstream", 
                #"activity_movingtime", 
                #"activity_name",
                "activity_timestream", 
                #"activity_type", 
                "activity_wattstream"]

n_workout = 21170
lines_per_workout = 194

# replace X with the actual line number where "power_is_from_hr" is expected to be (0-indexed)
power_is_from_hr_line = 40
activity_haspower_line = 6
activity_hrstream_line = 7
activity_timestream_line = 11
activity_wattstream_line = 13
#alpha_pstream_line = 16
alpha_stream_line = 17
artifact_percentage_line = 19
#ess_line = 24

# Initialize the data dictionary
data = {key: [] for key in desired_keys}
filtered_workout_count = 0
workout_count = 0

with open('cycling-activity-dump-2.jsonl') as file:
    while True:
        # read lines of a single workout
        workout_lines = [file.readline() for _ in range(lines_per_workout)]

        if len(workout_lines) < lines_per_workout:
            # end of file reached
            break

        # Read the necessary lines and strip whitespace
        power_is_from_hr_line_content = workout_lines[power_is_from_hr_line].strip()
        activity_haspower_line_content = workout_lines[activity_haspower_line].strip()
        activity_hrstream_line_content = workout_lines[activity_hrstream_line].strip()
        activity_wattstream_line_content = workout_lines[activity_wattstream_line].strip()
        activity_timestream_line_content = workout_lines[activity_timestream_line].strip()
        #alpha_pstream_line_content = workout_lines[alpha_pstream_line].strip()
        alpha_stream_line_content = workout_lines[alpha_stream_line].strip()
        artifact_percentage_line_content = workout_lines[artifact_percentage_line].strip()
        #ess_line_content = workout_lines[ess_line].strip()


        # Check the conditions:
        # MUST HAVE: ess, power, power not from HR
        # do NOT necessarily need alpha. Also without alpha, we need ess
        # BIG ASSUMPTION on ESS: this is always computed using power (or always HR)
        if (
                '"power_is_from_hr": false' in power_is_from_hr_line_content
                and 'null' not in activity_haspower_line_content
                and 'null' not in activity_hrstream_line_content
                and 'null' not in activity_wattstream_line_content
                and 'null' not in activity_timestream_line_content
                # and 'null' not in ess_line_content
                and 'null' not in alpha_stream_line_content
                and 'null' not in artifact_percentage_line_content
                # and 'null' not in alpha_pstream_line_content
                # and float(artifact_percentage_line_content.split(":")[1].replace(",", "").strip()) < 5
            ):
            filtered_workout_count += 1
            # print(artifact_percentage_line_content)
            # Join the workout lines to form a single JSON string
            workout_text = "".join(workout_lines)

            try:
                # Try to parse the JSON string to a dictionary
                workout_json = json.loads(workout_text)

                # If parsing was successful, extract and save the desired keys
                for key in desired_keys:
                    data[key].append(workout_json.get(key, np.nan))

            except json.JSONDecodeError:
                print(f"Could not parse workout data at workout {workout_count}")

        workout_count += 1
        if workout_count % 2117 == 0:
            print(f"workout number: {workout_count}")

        if workout_count >= n_workout:
            break

print(f"Number of workouts processed: {workout_count}")
print(f"Valid workouts: {filtered_workout_count}")

# Convert the data dictionary to a pandas DataFrame
df_raw = pd.DataFrame(data)
#%%
# filter only those with decent artifact %, otherwise HR kinda sucks
df_raw = df_raw[df_raw['artifact_percentage'] < 5]
print(df_raw.info())
#%%
# Checking the unique types of each of the three columns in your actual dataset
unique_types_hr = df_raw['activity_hrstream'].apply(lambda x: type(x)).unique()
unique_types_time = df_raw['activity_timestream'].apply(lambda x: type(x)).unique()
unique_types_watt = df_raw['activity_wattstream'].apply(lambda x: type(x)).unique()
print(unique_types_hr) 
print(unique_types_time)
print(unique_types_watt)
# Filtering rows where any of the three columns has a non-string type
non_string_rows = df_raw[(df_raw['activity_hrstream'].apply(lambda x: type(x) != str)) |
                         (df_raw['activity_timestream'].apply(lambda x: type(x) != str)) |
                         (df_raw['activity_wattstream'].apply(lambda x: type(x) != str))]
# Printing the first 2 such instances
print(non_string_rows.head(2))
#%%
# The data is obtained from JSONL, so they are objects, 
# turn them into lists: JSON LOAD CRASHES the system, use eval !!!

import ast
def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s  # Return original value if conversion fails

df_raw['activity_hrstream'] = df_raw['activity_hrstream'].apply(string_to_list)
print(df_raw.info())
#%%
df_raw['activity_timestream'] = df_raw['activity_timestream'].apply(string_to_list)
#%%
df_raw['activity_wattstream'] = df_raw['activity_wattstream'].apply(string_to_list)
print(df_raw['activity_hrstream'].apply(lambda x: type(x)).unique())
print(df_raw['activity_timestream'].apply(lambda x: type(x)).unique())
print(df_raw['activity_wattstream'].apply(lambda x: type(x)).unique())
#%%
same_lenghts = df_raw.apply(lambda x: len(x['activity_hrstream']) == len(x['activity_timestream']) == len(x['activity_wattstream']), axis=1)
print(len(same_lenghts)/len(df_raw))
#%%
'''
Since they all have the same length, as expected:
0. Check that timestream indeed has ALL numbers, no nans or none: OK!
1. Consider only the first 30 mins = 1800 seconds
2. Chop the 3 streams accordingly, stopping at the same index
'''
import math
def check_list_validity(lst):
    for element in lst:
        if element is None or not isinstance(element, (int, float)) or not math.isfinite(element):
            print(f"Invalid element found: {element}")
            return False
    return True
print(df_raw['activity_timestream'].apply(check_list_validity).all())
#%%
def chop_lists(timestream, hrstream, wattstream, end_time):
    """
    Function to chop lists based on the sum of timestream entries.

    Parameters:
    timestream (list): List of time values.
    hrstream (list): List of heart rate values.
    wattstream (list): List of watt values.
    end_time (int or float): Threshold for sum of time values.

    Returns:
    tuple: Chopped lists (timestream, hrstream, wattstream)
    """
    # Check if the lists are empty or not of the same length
    if not timestream or not hrstream or not wattstream:
        return (np.nan, np.nan, np.nan)
    if len(timestream) != len(hrstream) or len(timestream) != len(wattstream):
        return (np.nan, np.nan, np.nan)
    # Identifying the index where the sum of timestream values exceeds x
    timestream_new = []
    cumulative_sum = 0
    for index, value in enumerate(timestream):
        cumulative_sum += value
        timestream_new.append(cumulative_sum)
        if cumulative_sum > end_time:
            # Chop the lists up to the identified index
            if index != 0:
                return (timestream_new[:index], hrstream[:index], wattstream[:index])
            else:
                return (np.nan, np.nan, np.nan)
    # If the sum never exceeds x, return the full lists
    return (timestream_new, hrstream, wattstream)

# Apply the function to chop each list to 30 mins
df_raw[['edited_timestamps', 'edited_hrstream', 'edited_wattstream']] = df_raw.apply(
    lambda row: chop_lists(row['activity_timestream'], row['activity_hrstream'], row['activity_wattstream'], 30*60), 
    axis=1, result_type='expand'
)
#%%
# df_raw.drop('edited_timestream', axis=1, inplace=True)
# df_raw['activity_date'] = pd.to_datetime(df_raw['activity_date'])
print(df_raw.info())
print(df_raw[~df_raw['edited_timestamps'].apply(lambda x: isinstance(x, list)) |
             ~df_raw['edited_wattstream'].apply(lambda x: isinstance(x, list)) |
             ~df_raw['edited_hrstream'].apply(lambda x: isinstance(x, list))]
      [['activity_timestream', 'edited_timestamps', 'edited_wattstream', 'edited_hrstream']])

#%%

"""
    Function to identify entries in hrstream, and wattstream that are not numbers,
    and collect tuples of corresponding data points from 
    timestream, timestamps, hrstream, and wattstream.

    Parameters:
    timestream (list): List of time differences.
    timestamps (list): List of timestamps.
    hrstream (list): List of heart rate values.
    wattstream (list): List of watt values.

    Returns:
    list: List of tuples for the relevant intervals.
"""
no_number_set = set() 

def check_not_numbers(row):
    timestream = row['activity_timestream']
    timestamps = row['edited_timestamps']
    hrstream = row['edited_hrstream']
    wattstream = row['edited_wattstream']

    tuples_list = []
    
    
    # Check if lists are of the same length
    if (
        not isinstance(timestamps, list) 
        or not isinstance(hrstream, list) 
        or not isinstance(wattstream, list)
        or not len(timestamps) == len(hrstream) == len(wattstream)
        ):
        # print("Error: Lists are not of the same length or are NaNs")
        return np.nan 

    # Identify positions where wattstream or hrstream values are not numbers
    for i, _ in enumerate(hrstream):
        if not (isinstance(hrstream[i], (int, float)) and math.isfinite(hrstream[i])) or \
           not (isinstance(wattstream[i], (int, float)) and math.isfinite(wattstream[i])):
            if not (isinstance(hrstream[i], (int, float)) and math.isfinite(hrstream[i])):
                no_number_set.add(hrstream[i])
            if not (isinstance(wattstream[i], (int, float)) and math.isfinite(wattstream[i])):
                no_number_set.add(wattstream[i])

            tuples_list.append((timestream[i], timestamps[i], hrstream[i], wattstream[i]))
    
    return tuples_list


df_raw['not_numbers'] = df_raw.apply(check_not_numbers, axis=1)
print(df_raw['not_numbers'].head())
print(no_number_set)

#%%
# we have to insert NaNs in place of Nones here in the edited streams
import numpy as np
import math

# Function to replace non-finite numbers with np.nan in a list
def replace_with_nan(lst):
    if isinstance(lst, list):
        return [x if isinstance(x, (int, float)) and math.isfinite(x) else np.nan for x in lst]

# Apply this function to each of the specified columns
df_raw['edited_hrstream'] = df_raw['edited_hrstream'].apply(replace_with_nan)
df_raw['edited_wattstream'] = df_raw['edited_wattstream'].apply(replace_with_nan)
#%%
def average_excluding_none_nan(lst):
    if isinstance(lst, list):
        # Filter out None and NaN values
        filtered_lst = [x for x in lst if x is not None and not np.isnan(x)]
        if filtered_lst:
            return np.mean(filtered_lst)
    return np.nan
df_raw['avg_hr'] = df_raw['activity_hrstream'].apply(average_excluding_none_nan)
df_raw['avg_power'] = df_raw['activity_wattstream'].apply(average_excluding_none_nan)
#%%
# Plot timeseries, starting 5 minutes after the first timestamp with
# HR not zero
'''
In order to define the roughtness/smoothness of the hr vs power data, use volatility:
volatility = sd(differences of consecutive/other points in the series)
Smoothen out power (in symmetrical way) until volatility is around the same as HR volatility ?
'''

import matplotlib.pyplot as plt
import numpy as np
import os

unique_user_ids = df_raw['user_id'].unique()
n_rows = 2
delta_begin = 5*60 # In order to prevent issues with HR monitor at the beginning
time_f = 19*60

def find_first_non_zero(lst):
    return next((i for i, x in enumerate(lst) if x != 0 and not np.isnan(x)), np.nan)

# scale invariant volatilities, based on rel difference of consecutive points 
# OR based on rel deviations form the average
def volatility(lst):
    if not isinstance(lst, list) or len(lst) < 2:
        return np.nan
    # Calculate the percentage differences between consecutive elements, 
    # ignoring pairs with NaN
    percent_diffs = [(lst[i] - lst[i-1]) / lst[i-1] * 100 for i in range(1, len(lst))
                     if not math.isnan(lst[i]) and not math.isnan(lst[i-1]) and lst[i-1] != 0]
    if len(percent_diffs) < 2:
        return np.nan
    # Compute the standard deviation of these differences
    mean_diff = sum(percent_diffs) / len(percent_diffs)
    squared_diffs = [(d - mean_diff)**2 for d in percent_diffs]
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)
    std_dev = variance**0.5
    return std_dev

def mean_based_volatility(lst):
    if not isinstance(lst, list) or len(lst) < 2:
        return np.nan
    filtered_lst = [x for x in lst if not math.isnan(x)]
    if not isinstance(filtered_lst, list) or len(filtered_lst) < 2:
        return np.nan
    mean_value = np.mean(filtered_lst)
    deviations = [(x - mean_value) / mean_value * 100 for x in filtered_lst]
    # Compute the standard deviation of these differences
    mean_diff = sum(deviations) / len(deviations)
    squared_diffs = [(d - mean_diff)**2 for d in deviations]
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)
    std_dev = variance**0.5
    return std_dev

def smoothening(lst, timestamp_lst, delta_t, symmetric=False):
    '''
    This function smoothens the lst using the average of the points inside the
    time interval = [-delta_t, +delta_t] around each timestamp
    '''
    if not isinstance(lst, list) or not isinstance(timestamp_lst, list):
        return np.nan
    smoothed_lst = []
    for i, current_timestamp in enumerate(timestamp_lst):
        # Find indices of points within the time interval [-delta_t, 0/delta_t] around the current timestamp
        end_timestamp = current_timestamp
        if symmetric:
            end_timestamp = current_timestamp + delta_t
        relevant_indices = [j for j, t in enumerate(timestamp_lst) 
                            if current_timestamp - delta_t <= t <= end_timestamp]
        # Calculate the average of values in lst corresponding to these indices
        if relevant_indices:
            avg_value = np.nanmean([lst[j] for j in relevant_indices])
            smoothed_lst.append(avg_value)
        else:
            smoothed_lst.append(lst[i])  # If no relevant points are found, use the original value
    return smoothed_lst


# Iterate over each user ID
for user_id in unique_user_ids:
    # Filter the DataFrame for the current user ID and select the first 10 rows
    user_df = df_raw[df_raw['user_id'] == user_id].head(n_rows)
    for index, row in user_df.iterrows():
        # timestream = row['activity_timestream']
        timestamps = row['edited_timestamps']
        hrstream = row['edited_hrstream']
        wattstream = row['edited_wattstream']
        avg_hr = row['avg_hr']
        avg_power = row['avg_power']
        if (
        not isinstance(timestamps, list) 
        or not isinstance(hrstream, list) 
        or not isinstance(wattstream, list)
        or np.isnan(avg_hr)
        or np.isnan(avg_power)
        or not len(timestamps) == len(hrstream) == len(wattstream)
        ):
            continue
        
        i_hr = i_hr = find_first_non_zero(hrstream)
        if np.isnan(i_hr):
            continue
        i_begin = next((i for i, t in enumerate(timestamps) \
                        if t > (timestamps[i_hr] + delta_begin)), np.nan)
        i_end = next((i for i, x in enumerate(timestamps) \
                      if x >= min(time_f, timestamps[-1])), np.nan)
        if np.isnan(i_begin) or np.isnan(i_end):
            continue
        times, powers, hrs = timestamps[i_begin:i_end], wattstream[i_begin:i_end], hrstream[i_begin:i_end]
        # smoothen the power in a symmetric way
        powers_smooth1 = smoothening(powers, times, 60.1, symmetric=True)
        powers_smooth2 = smoothening(powers, times, 60.1)

        power_volatility = volatility(powers)
        hr_volatility = volatility(hrs)
        mean_based_power_volatility = mean_based_volatility(powers)
        mean_based_hr_volatility = mean_based_volatility(hrs)

        powers_smooth1_volatility = volatility(powers_smooth1)
        mean_based_powers_smooth1_volatility = mean_based_volatility(powers_smooth1)

        powers_smooth2_volatility = volatility(powers_smooth2)
        mean_based_powers_smooth2_volatility = mean_based_volatility(powers_smooth2)


        # Normalizing time series (ignoring NaNs)
        normalized_power = [(p/avg_power) for p in powers]
        normalized_hr = [(h/avg_hr) for h in hrs]
        normalized_power_smooth1 = [(p/avg_power) for p in powers_smooth1]
        normalized_power_smooth2 = [(p/avg_power) for p in powers_smooth2]

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(times, normalized_power, 
            label=(f'Normalized Power, volatility: {power_volatility:.2g}, '
                f'mean based: {mean_based_power_volatility:.2g}'),
            color='blue')
        plt.plot(times, normalized_hr, 
            label=(f'Normalized HR, volatility {hr_volatility:.2g}, '
                f'mean based: {mean_based_hr_volatility:.2g}'), 
            color='green')
        plt.plot(times, normalized_power_smooth1, 
            label=(f'Normalized Power, volatility: {powers_smooth1_volatility:.2g}, '
                   f'mean based: {mean_based_powers_smooth1_volatility:.2g}'), 
            color='orange')
        plt.plot(times, normalized_power_smooth2, 
            label=(f'Normalized Power, volatility: {powers_smooth2_volatility:.2g}, '
                   f'mean based: {mean_based_powers_smooth2_volatility:.2g}'), 
            color='purple')
        plt.title(f'User ID: {user_id}, Workout: {index}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Values')
        plt.legend()

        # Save plot
        plt.savefig(f'hr_lags/p_hr_a_{user_id}_{index}.png')
        plt.close()

#%%
# check the derivatives (differences of consecutive points)
from scipy.stats import spearmanr

n_rows = 3
t_lags = range(0, 50)

# Iterate over each user ID
for user_id in unique_user_ids:
    # Filter the DataFrame for the current user ID and select the first 10 rows
    user_df = df_raw[df_raw['user_id'] == user_id].head(n_rows)        
    user_df = user_df.sample(n=min(n_rows,len(user_df)))

    for index, row in user_df.iterrows():
        timestamps = row['edited_timestamps']
        hrstream = row['edited_hrstream']
        wattstream = row['edited_wattstream']
        avg_hr = row['avg_hr']
        avg_power = row['avg_power']
        if (
        not isinstance(timestamps, list) 
        or not isinstance(hrstream, list) 
        or not isinstance(wattstream, list)
        or np.isnan(avg_hr)
        or np.isnan(avg_power)
        or not len(timestamps) == len(hrstream) == len(wattstream)
        ):
            continue
        i_hr = i_hr = find_first_non_zero(hrstream)
        if np.isnan(i_hr):
            continue
        i_begin = next((i for i, t in enumerate(timestamps) \
                        if t > (timestamps[i_hr] + delta_begin)), np.nan)
        i_end = next((i for i, x in enumerate(timestamps) \
                      if x >= min(time_f, timestamps[-1])), np.nan)
        if np.isnan(i_begin) or np.isnan(i_end):
            continue

        times, powers, hrs = timestamps[i_begin:i_end], wattstream[i_begin:i_end], hrstream[i_begin:i_end]
        normalized_power = [p/avg_power for p in powers]
        normalized_hr = [h/avg_hr for h in hrs]
        hr_derivatives = [
                normalized_hr[i]/normalized_hr[i-1] - 1
                if (not np.isnan(normalized_hr[i]) 
                    and not np.isnan(normalized_hr[i-1]) 
                    and normalized_hr[i-1] != 0)
                else np.nan 
                for i in range(1, len(normalized_hr))
            ]

        normalized_power_smooth = {}  # Dictionary to store smoothed data
        correlations = {}
        for t_lag in t_lags:
            smoothed_powers = smoothening(powers, times, t_lag)
            normalized_power_smooth[t_lag] = [p/avg_power for p in smoothed_powers]

            # Calculate derivatives
            power_smooth_derivatives = [
                smoothed_powers[i]/smoothed_powers[i-1] - 1
                if (not np.isnan(smoothed_powers[i]) 
                    and not np.isnan(smoothed_powers[i-1]) 
                    and smoothed_powers[i-1] != 0)
                else np.nan 
                for i in range(1, len(smoothed_powers))
            ]
            
            # corr_all, _ = spearmanr(hr_derivatives, power_smooth_derivatives)
            # Compute Spearman correlation only for points with abs(power_smooth_derivatives) >= 0.1
            filtered_indices = [i for i, val in enumerate(power_smooth_derivatives) if abs(val) >= 0.05]
            if len(filtered_indices) > 2:  # Need at least two points to compute correlation
                filtered_hr_derivatives = [hr_derivatives[i] for i in filtered_indices]
                filtered_power_derivatives = [power_smooth_derivatives[i] for i in filtered_indices]
                corr, p_value = spearmanr(filtered_hr_derivatives, filtered_power_derivatives)
            else:
                corr, p_value = np.nan, np.nan  # Set to NaN if not enough data points
            
            correlations[t_lag] = [corr, p_value, len(filtered_indices)]

        # plot streams and correlations
        if all(np.isnan(correlation[0]) for correlation in correlations.values()) or \
            max(correlation[0] for correlation in correlations.values() if not np.isnan(correlation[0])) < 0.4:
            continue
        
        max_t_lag = max(correlations, key=lambda x: correlations[x][0] if not np.isnan(correlations[x][0]) else float('-inf'))

        plt.figure(figsize=(12, 8))
        plt.plot(times, normalized_power, label='Normalized Power', color='grey', linestyle='--')
        plt.plot(times, normalized_power_smooth[max_t_lag], 
            label=(f'Power smooth {max_t_lag} s (normalized)'),
            color='blue')
        plt.plot(times, normalized_hr, 
            label=(f'HR (normalized)'), 
            color='red')
        plt.title(f'p vs hr with lag = {max_t_lag}s (corr: {correlations[t_lag]}) - User ID: {user_id}, Workout: {index}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(f'hr_lags/corr_lags_{user_id}_{index}_2.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        correlation_values = [corr[0] for corr in correlations.values()]
        plt.plot(correlations.keys(), correlation_values, marker='o', linestyle='-')
        plt.xlabel('t_lag')
        plt.ylabel('Correlation')
        plt.title(f'Correlation vs. Time Lag - User ID: {user_id}, Workout: {index}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'hr_lags/corr_lags_{user_id}_{index}.png')
        plt.close()


#%%
import warnings
from scipy.stats import spearmanr

# Define the lags (in terms of number of steps, given 5-second spacing)


# Iterate over each user ID
for user_id in unique_user_ids:
    user_df = df_raw[df_raw['user_id'] == user_id]
    user_df = restrict to 5 random rows of user_df
    for index, row in user_df.iterrows():




# Function to calculate correlation with power at a given lag
def compute_correlation(tuples_list, lag, quantity, time_start=None, duration=None, excluding_p0=False):
    tuples = select_tuples(tuples_list, time_start, duration, excluding_p0)

    if quantity not in quantity_indices:
        raise ValueError("Invalid quantity requested. Please indicate 'hr' or 'alpha1'.")

    power = [t[1] for t in tuples]
    hr_metric = [t[quantity_indices[quantity]] for t in tuples]

    if lag >= len(hr_metric):
        return np.nan

    # Shift HR backward by the specified lag
    shifted_hr_metric = np.full_like(hr_metric, np.nan)  # array of NaNs
    if lag > 0:
        shifted_hr_metric[:-lag] = hr_metric[lag:]  # Fill with shifted values
    else:
        shifted_hr_metric = hr_metric  # No shift for lag = 0

    # NOW: Pair and filter out NaN and p=0 pairs
    paired_data = [(p, h) for p, h in zip(power, shifted_hr_metric) if not np.isnan(p) and not np.isnan(h) and not p==0]

    # Ensure there are at least 30% valid pairs, to get good stats
    if len(paired_data) < 0.3*len(tuples):
        return np.nan

    # Separate the pairs for correlation computation
    power_clean, shifted_clean = zip(*paired_data)

    # Catching warnings during correlation computation
    # with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", RuntimeWarning)
    correlation, _ = spearmanr(power_clean, shifted_clean)

    return correlation

# Iterate over each row and calculate correlations
t_start = 5*60
duration_interval = 20*60
for lag_time in lags:
    df[f'corr_lag{lag_time*5}'] = df.apply(
    lambda row: compute_correlation(
        tuples_list=row['tuples_edit'],
        lag=lag_time,
        quantity='alpha1',  # Replace with the desired quantity
        time_start=t_start,
        duration=duration_interval
    ), axis=1
)
#%%
print(df.info())
print(df.iloc[0])
#%%
def compute_pct_below_thr(tuples_list, alpha_thr):
    if not tuples_list:  # Check for an empty list
        return np.nan
    count_below = sum(1 for t in tuples_list if t[2] < alpha_thr)
    return count_below / len(tuples_list)

alpha_thr = 1
df['pct_below_a_thr'] = df.apply(
    lambda row: compute_pct_below_thr(tuples_list=row['tuples_edit'], alpha_thr=alpha_thr), axis=1
)
print(df.info())
#%%
# Plotting for the first 20 rows of each user_id
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming df is your DataFrame

# Get the unique user IDs
unique_user_ids = df['user_id'].unique()
n_rows = 40

# Iterate over each user ID
for user_id in unique_user_ids:
    # Filter the DataFrame for the current user ID and select the first 10 rows
    user_df = df[df['user_id'] == user_id]
    if len(user_df) < n_rows:
        n_rows = len(user_df)

    user_df = user_df.head(n_rows)

    # Create a figure for the plot
    plt.figure(figsize=(12, 8))

    # Define correlation columns
    correlation_columns = [col for col in df.columns if col.startswith('corr_lag')]

        # Plot each row's correlations with a different color
    for index, row in user_df.iterrows():
        color = 'red' if row['pct_below_a_thr'] > 0.5 else 'blue'
        plt.plot(correlation_columns, row[correlation_columns], marker='o', label=f'Row {index}', color=color)

    # Adding plot title, labels, and legend
    plt.title(f'Correlation for User ID {user_id}')
    plt.xlabel('Lag (in seconds)')
    plt.ylabel('Correlation')
    plt.xticks(ticks=np.arange(len(correlation_columns)), labels=[col.replace('corr_lag', '') for col in correlation_columns])
    plt.legend()

    plt.savefig(f'lags_{user_id}.png')
    plt.close()

