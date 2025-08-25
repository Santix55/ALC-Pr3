import json
import pandas as pd

# Read the JSON files
with open('lab3_materials/golds_task3_exist2025/SMG_XMLRobertaAll2.json', 'r') as file1, open('lab3_materials/golds_task3_exist2025/SMG_OnlyEN-Roberta.json', 'r') as file2:
    data1 = json.load(file1)
    data2 = json.load(file2)

# Convert the JSON in dataframes
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Show the dataframes
print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Overwrite the data of df1 with the entries of df2
df1.set_index('id', inplace=True)
df2.set_index('id', inplace=True)
df1 = df2.combine_first(df1).reset_index()

print("\nResult:")
print(df1)

# Convert into a JSON again
df1.reset_index(inplace=True)
df1.to_json('lab3_materials/golds_task3_exist2025/SMG_XMLRobertaALL+RobertaEN.json', orient='records')