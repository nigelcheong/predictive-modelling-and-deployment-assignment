import pandas as pd

# LOAD DATA
df = pd.read_csv('data/zomato_df_final_data.csv')

# PRE-PROCESSING
df_clean = df.dropna(subset=['rating_number']) # Create subset of the dataset where rating_number is not NA

# List of numeric columns to impute
cols_to_impute = ['cost', 'cost_2']   # add more columns here if needed

for col in cols_to_impute:
    median_value = df_clean[col].median()
    df_clean.loc[df_clean[col].isna(), col] = median_value

df_clean.loc[df_clean['type'].isna(), 'type'] = [['Unknown']]

# Impute latitude by suburb
df_clean.loc[df_clean['lat'].isna(), 'lat'] = (
    df_clean.groupby('subzone')['lat'].transform('median')
)

# Impute longitude by suburb
df_clean.loc[df_clean['lng'].isna(), 'lng'] = (
    df_clean.groupby('subzone')['lng'].transform('median')
)

# Create subzone_clean safely with .loc
df_clean.loc[:, 'subzone_clean'] = (
    df_clean['subzone'].str.split(',').str[-1].str.strip()
)

# Impute latitude
df_clean.loc[df_clean['lat'].isna(), 'lat'] = (
    df_clean.groupby('subzone_clean')['lat'].transform('median')
)

# Impute longitude
df_clean.loc[df_clean['lng'].isna(), 'lng'] = (
    df_clean.groupby('subzone_clean')['lng'].transform('median')
)

# Save preprocessed dataset
df_clean.to_csv("data/preprocessed.csv", index=False)
print("Preprocessing complete → data/preprocessed.csv")
