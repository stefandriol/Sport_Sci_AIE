# %%
import numpy as np
import pandas as pd
import json
import re
 # %%
''' VOICES  for each line of JSON '''

k = 1
k2 = 192
extracted_strings = []

with open('cycling-activity-dump-2.jsonl') as file:
    for i, line in enumerate(file, start=0): # Start enumerating from 0
        if i < k: # Skip the
            continue
        if i > k2: # Stop after the 10th line
            break

        # Extract the first string between quotes from the line
        match = re.search(r'"(.*?)"', line)
        if match:
            extracted_strings.append(match.group(1))

# Now, extracted_strings contains all the strings extracted from the lines between k and k2
for string in extracted_strings:
    print(string)

# %% 
''' TOT number of lines (WORKOUTS) '''

with open('cycling-activity-dump-2.jsonl') as file:
    line_count = sum(1 for line in file if line.strip())
    workout_count = line_count / 194

print(f"Total number of workouts: {workout_count}")
# %%

''' GET DATA '''

'''
desired_keys = ["user_id", "activity_date", "activity_type", "plot_data", "ess", "activity_movingtime", 
"artifact_percentage", "alpha_stream", "alpha_pstream", "alpha_stream_samplerate" , "alpha_hrstream", 
"d_alpha", "source", "avalpha", "activity_avhr", "activity_avwatts", 
"activity_timestream", "activity_hrstream", "from_hr_cverror_percent", "hr_alpha_end", 
"hr_alpha_end_cluster", "hr_alpha_thr", "hr_alpha_thr_cluster", "is_erg_mode", "iss", "watts_alpha_end", 
"watts_alpha_end_cluster", "watts_alpha_thr", "watts_alpha_thr_cluster"]
'''
desired_keys = ["user_id", "activity_date", 
                "plot_data", 
                "artifact_percentage", 
                # "activity_hrstream", 
                # "activity_wattstream",
                # "hr_alpha_end", 
                # "hr_alpha_end_cluster", 
                # "hr_alpha_thr", 
                # "hr_alpha_thr_cluster", 
                # "watts_alpha_end", 
                # "watts_alpha_end_cluster", 
                # "watts_alpha_thr", 
                #"watts_alpha_thr_cluster"
                ]

n_workout = 21170

lines_per_workout = 194
power_is_from_hr_line = 40
# replace X with the actual line number where "power_is_from_hr" is expected to be (0-indexed)
activity_haspower_line = 6
# alpha_pstream_line = 16
# alpha_stream_line = 17
# artifact_percentage_line = 19
# ess_line = 24

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
        # alpha_pstream_line_content = workout_lines[alpha_pstream_line].strip()
        # alpha_stream_line_content = workout_lines[alpha_stream_line].strip()
        # artifact_percentage_line_content = workout_lines[artifact_percentage_line].strip()
        # ess_line_content = workout_lines[ess_line].strip()


        # Check the conditions:
        # MUST HAVE ess, power, power not from HR
        # do NOT necessarily need alpha. Also without alpha, we need ess
        # BIG ASSUMPTION on ESS: this is always computed using power (or always HR)        
        if (
                '"power_is_from_hr": false' in power_is_from_hr_line_content
                and 'null' not in activity_haspower_line_content
                # and 'null' not in ess_line_content
                # and 'null' not in alpha_stream_line_content
                # and 'null' not in artifact_percentage_line_content
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
                    data[key].append(workout_json.get(key, None))

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
# %%
import re
import ast

problematic_indices = set()

def keys_present(data_str, keys):
    return all(k in data_str for k in keys)

def extract_row_data(row):
    # If 'artifact_percentage' is empty, don't process this row
    if pd.isna(row['artifact_percentage']):
        return row

    data_str = row['plot_data']
    if not isinstance(data_str, str):  # Ensure the input data is a string
        problematic_indices.add(row.name)
        return row

    required_keys = ['time', 'power', 'alpha1']
    if not keys_present(data_str, required_keys):
        problematic_indices.add(row.name)
        return row

    for key in required_keys + ['heartrate']:
        # Using regex to capture the content related to the specified key
        pattern = re.compile(r"'{}'\s*:\s*(\[.*?\])".format(key))
        match = pattern.search(data_str)
        # If there's a match, attempt conversion only on that part
        if match:
            value_str = match.group(1)
            # Replace lowercase 'nan' and 'NaN' with a placeholder string
            value_str_modified = value_str.replace('nan', '"PLACEHOLDER_NAN"').replace('NaN', '"PLACEHOLDER_NAN"')
            try:
                value = ast.literal_eval(value_str_modified)
                # Convert the placeholder string back to float('nan')
                value = [float('nan') if x == 'PLACEHOLDER_NAN' else x for x in value]
                row[key] = value
            except Exception as e:
                print("Error:", e)
                print(f"Culprit data for key '{key}':", value_str)
                raise
    return row

# Process the DataFrame row-wise
df_raw = df_raw.apply(extract_row_data, axis=1)
print(len(problematic_indices))
print(df_raw.info())

# %%
''' plot data may be missing altogether or containing only some keys (other are NaN) 
'''
# check plot_data keys for one problematic index to check that indeed there is something missing
import re

unique_keys = set()
pattern = re.compile(r"'(.*?)'")

problematic_list = list(problematic_indices)
data_str = df_raw['plot_data'][problematic_list[0]]
keys = pattern.findall(data_str)
unique_keys.update(keys)
print(unique_keys)
print(df_raw['artifact_percentage'][problematic_list[0]])
# %%
''' DROP PLOT DATA column to free space '''
df_raw.drop('plot_data', axis=1, inplace=True)
# %%
# check the types of time, power etc columns

columns_to_check = ['time', 'power', 'alpha1', 'heartrate']

unique_types = {}

for col in columns_to_check:
    types_in_col = df_raw[col].apply(lambda x: type(x)).unique()
    unique_types[col] = types_in_col

for col, types in unique_types.items():
    print(f"Unique types in '{col}': {types}")
# %%
'''
# check indeed floats are nans
def print_float_instances(column_name, num_instances=5):
    float_rows = df_raw[df_raw[column_name].apply(lambda x: isinstance(x, float))]
    print(f"\nShowing {num_instances} float instances for column '{column_name}':")
    print(float_rows[[column_name]].head(num_instances))

columns_to_check = ['time', 'power', 'alpha1', 'heartrate']

for col in columns_to_check:
    print_float_instances(col)
'''
# %%
''' Lists may contain NaNs
'''
# count NaNs in each column THAT IS A LIST

import math
def count_nans(value):
    if isinstance(value, list):
        return sum(1 for item in value if isinstance(item, float) and math.isnan(item))
    return 0  # Return 0 otherwise

# Count NaNs in each column and print results
for col in ['time', 'power', 'alpha1', 'heartrate']:
    nans_count = df_raw[col].apply(count_nans).sum()
    print(f"Number of NaNs inside lists in '{col}': {nans_count}")
# %%
''' RESTRICT TO MEANINGFUL DATA: lists with same lenght '''

def check_lists_same_length(row):
    # Extract the values for the columns of interest
    cols = [row['time'], 
            row['power'], 
            row['alpha1'], 
            row['heartrate']
            ]
    
    # Check if all elements are lists
    if all(isinstance(col, list) for col in cols):
        # Extract lengths of each list
        lengths = [len(col) for col in cols]
        
        # Check if all lengths are the same
        return len(set(lengths)) == 1
    else:
        return False
    
df = df_raw[df_raw.apply(check_lists_same_length, axis=1)]
print(df_raw.info())
print(df.info())
# %%
''' Remove NaNs from lists '''
def clean_data(row):
    # Extract lists from the row
    time = row['time']
    power = row['power']
    alpha1 = row['alpha1']
    heartrate = row['heartrate']
    
    # Initialize new lists to hold cleaned data
    cleaned_time = []
    cleaned_power = []
    cleaned_alpha1 = []
    cleaned_heartrate = []
    
    # Iterate through each entry by its index
    for i in range(len(time)):
        # Check if any of the three lists have a NaN at the current index
        if pd.isna(power[i]) or pd.isna(alpha1[i]) or pd.isna(heartrate[i]) or pd.isna(time[i]):
            continue  # Skip this index if any list has a NaN value
        
        # If no NaN values, append the data to the cleaned lists
        cleaned_time.append(time[i])
        cleaned_power.append(power[i])
        cleaned_alpha1.append(alpha1[i])
        cleaned_heartrate.append(heartrate[i])
    
    # Update the row with cleaned lists
    row['time'] = cleaned_time
    row['power'] = cleaned_power
    row['alpha1'] = cleaned_alpha1
    row['heartrate'] = cleaned_heartrate
    
    return row

# Assuming df is your DataFrame
# Apply the clean_data function to each row
df = df.apply(clean_data, axis=1)
# %%
df.to_csv('data.csv', index=False)
# %%
