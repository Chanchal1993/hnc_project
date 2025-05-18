import pandas as pd
import os

# Paths
input_csv = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/csv/hnc_train_sample.csv"
output_dir = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/csv/"
output_csv = os.path.join(output_dir, "hnc_train_sample_preprocessed.csv")

# Load the dataset
data = pd.read_csv(input_csv)

# Clean column names
data.columns = data.columns.str.strip()

# Extract Patient IDs (exclude from preprocessing)
patient_ids = data['Patient #']
features = data.drop(columns=['Patient #'])

# Define mappings for known categorical columns (add more as needed)
categorical_mappings = {
    'gender': {'M': 0, 'F': 1},
    # Add more mappings if known
}

# Apply the mappings
for col, mapping in categorical_mappings.items():
    if col in features.columns:
        features[col] = features[col].map(mapping)

# Identify remaining categorical columns (object type)
remaining_categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
print(f"Remaining categorical columns: {remaining_categorical_cols}")

# Apply one-hot encoding to remaining categorical columns (excluding Patient #)
features = pd.get_dummies(features, columns=remaining_categorical_cols)

# Fill missing numeric values
features = features.fillna(features.mean())

# Recombine Patient IDs as the first column
processed_data = pd.concat([patient_ids, features], axis=1)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save preprocessed data
processed_data.to_csv(output_csv, index=False)
print(f"Preprocessed data saved to {output_csv}")
