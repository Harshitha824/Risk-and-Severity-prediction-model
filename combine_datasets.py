import os
import pandas as pd

BASE_DIR = os.path.join(os.getcwd(), "datasets")
output_path = os.path.join(BASE_DIR, "fever_master_dataset.csv")

dataframes = []
print(" Scanning all subfolders in:", BASE_DIR)

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            print(f"\n Loading {file_path} ...")
            loaded = False
            for enc in ["utf-8", "latin-1", "ISO-8859-1"]:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    if not df.empty:
                        df["source_dataset"] = os.path.basename(root)
                        dataframes.append(df)
                        print(f" Loaded {file} | Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                    else:
                        print(f" {file} is empty.")
                    loaded = True
                    break
                except Exception as e:
                    print(f" Error with encoding {enc}: {e}")
            if not loaded:
                print(f"Could not read {file_path}")

# Combine all valid dataframes
if dataframes:
    print("\n Combining datasets...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_df.dropna(how="all", inplace=True)

    combined_df.to_csv(output_path, index=False)
    print("\nCombined dataset created successfully!")
    print(f" Saved as: {output_path}")
    print(f" Total Rows: {combined_df.shape[0]} | Columns: {combined_df.shape[1]}")
else:
    print("\n No data could be merged! Please verify that your dataset folders contain valid CSVs.")
