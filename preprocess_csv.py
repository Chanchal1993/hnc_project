import pandas as pd

csv_path = "/Users/chanchalm/Documents/hnc_dataset/numpy_files/train/csv/hnc_train_sample.csv"
data = pd.read_csv(csv_path)

# Clean column names
data.columns = data.columns.str.strip()

# Extract Patient IDs (exclude from preprocessing)
patient_ids = data['Patient #']
features = data.drop(columns=['Patient #'])

# Convert only string categorical columns (dtype object and values are strings)
for col in features.columns:
    if features[col].dtype == 'object':
        # Check if the column contains strings (not numbers or mixed types)
        if features[col].apply(lambda x: isinstance(x, str)).all():
            features[col] = features[col].astype('category').cat.codes
            print(f"Converted column '{col}' to categorical codes.")

# Fill missing numeric values
features = features.fillna(features.mean())

# Recombine Patient IDs
processed_data = pd.concat([patient_ids, features], axis=1)

output_path = "/Users/chanchalm/Documents/hnc_dataset/numpy_files/train/csv/hnc_train_processed.csv"
processed_data.to_csv(output_path, index=False)

print(f"Preprocessed data saved to {output_path}")
