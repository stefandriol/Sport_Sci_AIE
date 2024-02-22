#%% 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
# %%
print(df.info())
df = df[df['artifact_percentage'] < 5]
print(df.info())
# %%
''' check types of columns '''
# List of columns to check
columns_to_check = ['time', 'power', 'alpha1', 'heartrate']

for column in columns_to_check:
    # Use a set comprehension to get unique types of all elements in the column
    unique_types = {type(x) for x in df[column]}
    print(f"Unique types in {column}: {unique_types}")

# %%
''' since they are string representations of lists, convert them '''
import ast

def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError) as e:
        # Print the string and the type of error
        print(f"Error converting string: {string}")
        print(f"Error type: {type(e).__name__}")
        # Handle the case where the string cannot be converted
        return np.nan  # or some appropriate value

# Apply this conversion to each column as needed
for column in columns_to_check:
    df[column] = df[column].apply(lambda x: convert_string_to_list(x) if isinstance(x, str) else x)
# %%
df['activity_date'] = pd.to_datetime(df['activity_date'])
# %%
''' getting rid of:
1. empty time lists (all lists have same lenght) 
2. p=0 workouts
'''
df = df[df['time'].apply(len) > 0]
df = df[df['power'].apply(sum) > 0]
print(df.info())
#%%
''' FUNCTIONS:
1 - Averages and effective averages 
2 - activity time and spin time and ratio 
'''
# compute the averages / effective averages for both parts of workout
def calculate_averages(row, column_name, exclude_zero_power=False):
    # 1. Check if the column_name is among the columns of row
    if column_name not in row:
        print(f"Column '{column_name}' not found. Available columns are: {list(row.keys())}")
        return np.nan

    # 2. Check if row[column_name] is a list and process accordingly
    column_data = row[column_name]
    if not isinstance(column_data, list):
        print(f'This is not a list to be averaged over: {column_data}')
        return np.nan
    
    if exclude_zero_power:
        if not isinstance(row['power'], list):
            print('Cannot exclude p=0 values: power is not a list:')
            print(row['power'])
            return np.nan
        if any(p == 0 for p in row['power']):
            # Filter out items where power is 0
            column_data = [value for value, power in zip(column_data, row['power']) if power != 0]

    # 3. Return the average of the filtered list, or np.nan if the list is empty
    if column_data:
        return np.mean(column_data).round(2)
    else:
        print(f'List of {column_name} is empty. Printing the entire row:')
        print(row)
        return np.nan


def calculate_activity_time(row, exclude_zero_power=False):
    # Check if 'time' is a list
    if not isinstance(row['time'], list):
        print('Time is not a list:')
        print(row['time'])
        return np.nan
    
    tot_time = max(row['time'])
    
    if exclude_zero_power:
        if not isinstance(row['power'], list):
            print('Cannot exclude p=0 values: power is not a list:')
            print(row['power'])
            return np.nan
        zero_power_count = sum(1 for p in row['power'] if p == 0)
        tot_time -= 5 * zero_power_count
    
    return tot_time

# Calculate the percentages of ACTIVITY below threshold
def calculate_a1_below_threshold(row, threshold):
    if not isinstance(row['alpha1'], list) or not isinstance(row['power'], list):
        print('Power or alpha1 is not a list, so we cannot compute the active alpha1 percentage below {threshold}')
        return np.nan
    
    alpha1_data = row['alpha1']
    # consider only ACTIVE timestamps
    if any(p == 0 for p in row['power']):
        alpha1_data = [value for value, power in zip(alpha1_data, row['power']) if power != 0]

    if len(alpha1_data) == 0:
        print('No active alpha1 data found in this row')
        print(row['alpha1'])
        print(row['power'])        
        return np.nan
    count_below_threshold = sum(1 for a1 in alpha1_data if a1 <= threshold)
    ratio = count_below_threshold / len(alpha1_data)
    return round(ratio, 3)  # Round the ratio to X decimal places after the decimal point

# %%
df['power_avg'] = df.apply(calculate_averages, args=('power', False), axis=1)
df['alpha1_avg'] = df.apply(calculate_averages, args=('alpha1', False), axis=1)
df['heartrate_avg'] = df.apply(calculate_averages, args=('heartrate', False), axis=1)
df['power_eff'] = df.apply(calculate_averages, args=('power', True), axis=1)
df['alpha1_eff'] = df.apply(calculate_averages, args=('alpha1', True), axis=1)
df['heartrate_eff'] = df.apply(calculate_averages, args=('heartrate', True), axis=1)
# %%
df['activity_time'] = df.apply(calculate_activity_time, args=(False,), axis=1)
df['spin_time'] = df.apply(calculate_activity_time, args=(True,), axis=1)
df['spin_ratio'] = df.apply(lambda row: row['spin_time'] / row['activity_time'] if row['activity_time'] != 0 else np.nan, axis=1)
# %%
a1_thresholds = [1, 0.75, 0.5]
for a1_threshold in a1_thresholds:
    column_name = f'alpha1_below_{a1_threshold}'
    df[column_name] = df.apply(lambda row: calculate_a1_below_threshold(row, a1_threshold), axis=1)
# %%
print(df.info())
# print(df[df['spin_time'] == df['activity_time']])
print(df[df['spin_ratio'].isna()])
# %%
df.to_csv('data_prep.csv', index=False)
# %%
