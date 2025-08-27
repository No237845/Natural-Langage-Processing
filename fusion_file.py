import pandas as pd

# Define the column names for each dataset
burkina_info_columns = ["Politique", "Economie", "Société", "Culture", "Sport", "Tech"]
le_faso_columns = ["Politique", "Société"]

# Load the CSV files without headers and assign column names
burkina_info = pd.read_csv('BurkinaInfo.csv', header=None, names=burkina_info_columns)
le_faso = pd.read_csv('LeFaso.csv', header=None, names=le_faso_columns)

# Print column names to verify
print("BurkinaInfo columns:", list(burkina_info.columns))
print("LeFaso columns:", list(le_faso.columns))

# Add missing columns to LeFaso with null values
missing_columns = set(burkina_info.columns) - set(le_faso.columns)
for col in missing_columns:
    le_faso[col] = None

# Ensure both DataFrames have the same column order
le_faso = le_faso[burkina_info.columns]

# Concatenate the datasets
fusion = pd.concat([burkina_info, le_faso], ignore_index=True)

# Save the merged file
fusion.to_csv('fichier_fusionne.csv', index=False)

print("Fusion completed successfully!")
print(f"Final dataset shape: {fusion.shape}")