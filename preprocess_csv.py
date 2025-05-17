import pandas as pd
import os

# Paths
input_csv = "/Users/chanchalm/Documents/hnc_dataset/numpy_files/train/csv/hnc_train_sample.csv"
output_dir = "/Users/chanchalm/Documents/hnc_dataset/numpy_files/train/csv/"
output_csv = os.path.join(output_dir, "hnc_train_sample_preprocessed.csv")

# Load the dataset
data = pd.read_csv(input_csv)

# Display column types to identify categorical columns
print("Data types before processing:\n", data.dtypes)

# Define mappings for known categorical columns
categorical_mappings = {
    'gender': {'M': 0, 'F': 1},
    # Add more mappings if known, e.g.
    # 'smoking_status': {'never': 0, 'former': 1, 'current': 2},
}

# Apply the mappings
for col, mapping in categorical_mappings.items():
    if col in data.columns:
        data[col] = data[col].map(mapping)

# Identify remaining categorical columns (object type)
remaining_categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
print(f"Remaining categorical columns: {remaining_categorical_cols}")

# Apply one-hot encoding to remaining categorical columns
data = pd.get_dummies(data, columns=remaining_categorical_cols)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save preprocessed data
data.to_csv(output_csv, index=False)
print(f"Preprocessed data saved to {output_csv}")
