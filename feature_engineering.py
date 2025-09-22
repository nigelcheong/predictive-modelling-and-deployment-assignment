import pandas as pd
import ast

# LOAD DATA
df_clean = pd.read_csv("data/preprocessed.csv")

# ENCODING
order = ["Poor", "Average", "Good", "Very Good", "Excellent"]
df_clean['rating_text_encoded'] = pd.Categorical(df_clean['rating_text'], categories=order, ordered=True).codes

df_clean.loc[:, 'cuisine'] = df_clean['cuisine'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# FEATURE ENGINEERING
# Create cuisine diversity feature
df_clean.loc[:, 'cuisine_diversity'] = df_clean['cuisine'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Create binary rating feature
df_clean.loc[:, 'rating_binary'] = df_clean['rating_text'].apply(
    lambda x: 1 if x in ['Good', 'Very Good', 'Excellent'] else 0
)

# Save engineered dataset
df_clean.to_csv("data/engineered.csv", index=False)
print("Feature engineering complete → data/engineered.csv")
