import pandas as pd
import numpy as np

# Step 1: Import the Data into a DataFrame
url = "/content/sample_data/BL-Flickr-Images-Book.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Step 2: Find and Drop Irrelevant Columns
# Let's assume 'Identifier', 'Edition Statement', and 'Corporate Author' are irrelevant for this example
irrelevant_columns = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Former owner', 'Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']
df.drop(columns=irrelevant_columns, inplace=True)

# Display the first few rows after dropping irrelevant columns
print(df.head())

# Step 3: Change the Index of the DataFrame
# Assuming 'Identifier' is a suitable unique index
df.set_index('Identifier', inplace=True)

# Display the DataFrame after setting a new index
print(df.head())

# Step 4: Tidy Up Fields (e.g., Date of Publication)
# Use a simple regular expression to clean the 'Date of Publication' field
df['Date of Publication'] = df['Date of Publication'].str.extract(r'(\d{4})').astype(float)

# Display the first few rows after tidying up the 'Date of Publication' field
print(df.head())

# Step 5: Combine str Methods with NumPy to Clean Columns
# For example, let's clean the 'Place of Publication' column
df['Place of Publication'] = np.where(df['Place of Publication'].str.contains('London'), 'London', df['Place of Publication'])
df['Place of Publication'] = np.where(df['Place of Publication'].str.contains('Oxford'), 'Oxford', df['Place of Publication'])

# Display the first few rows after cleaning the 'Place of Publication' column
print(df.head())