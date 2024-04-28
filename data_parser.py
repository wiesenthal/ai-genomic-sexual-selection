import pandas as pd
import json
import os

# Read Excel file
df = pd.read_csv('sex_quiz.csv')

# Iterate over each row
for index, row in df.iterrows():
    # Convert the row to a dictionary
    row_data = row.to_dict()
    # Create a separate JSON file for each row
    with open(f'row_{index}.json', 'w') as f:
        f.write(json.dumps(row_data, indent=4))

