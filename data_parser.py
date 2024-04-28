import pandas as pd

# Read Excel file
df = pd.read_excel('sex_quiz.xlsx')

# Convert DataFrame to JSON
json_data = df.to_json(orient='records')

# Print or save the JSON data
print(json_data)
