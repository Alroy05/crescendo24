import pandas as pd


csv_file_path = 'amazon_reviews.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Replace 'column_name' with the actual name of the column you want to concatenate
column_name = 'Review'

# Concatenate content from the specified column into a single string
concatenated_string = ' '.join(df[column_name].astype(str))

print('Concatenated String:')
print(concatenated_string)
