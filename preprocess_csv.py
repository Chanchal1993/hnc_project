import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Paths
input_csv = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/csv/hnc_train_sample.csv"
output_dir = "/Users/varun.t1/Documents/vat_base/hnc_project/numpy_files/train/csv/"
output_csv = os.path.join(output_dir, "hnc_train_sample_preprocessed.csv")

def bin_categories(series, min_freq=0.05):
    """
    Bin rare categories into 'Other' category if their frequency is below min_freq
    """
    value_counts = series.value_counts(normalize=True)
    return series.map(lambda x: x if value_counts[x] >= min_freq else 'Other')

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

# Apply the mappings for known categorical columns
for col, mapping in categorical_mappings.items():
    if col in features.columns:
        features[col] = features[col].map(mapping)

# Identify remaining categorical columns (object type)
remaining_categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
print(f"Remaining categorical columns: {remaining_categorical_cols}")

# Store mappings for CSV output
mappings_data = []

# Apply binning and label encoding to remaining categorical columns
label_encoders = {}
for col in remaining_categorical_cols:
    # First bin the categories
    features[col] = bin_categories(features[col])
    
    # Then apply label encoding
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le
    
    # Store mapping for CSV
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    for category, encoded_value in mapping.items():
        mappings_data.append({
            'Column': col,
            'Original_Value': category,
            'Encoded_Value': encoded_value
        })
    
    # Print the mapping for reference
    print(f"\nMapping for {col}:")
    print(mapping)

# Fill missing numeric values
features = features.fillna(features.mean())

# Recombine Patient IDs as the first column
processed_data = pd.concat([patient_ids, features], axis=1)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save preprocessed data
processed_data.to_csv(output_csv, index=False)
print(f"Preprocessed data saved to {output_csv}")

# Save the label encoders mapping to CSV
mapping_file = os.path.join(output_dir, "label_encoders_mapping.csv")
pd.DataFrame(mappings_data).to_csv(mapping_file, index=False)
print(f"Label encoders mapping saved to {mapping_file}")
