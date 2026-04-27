import os
import json
import pandas as pd

def normalize_table_name(name):
    """
    Normalizes a table (or destination table) name.
    - Strips extra spaces.
    - Removes commas.
    - Replaces spaces with underscores.
    - Converts to lowercase.
    - Adds a leading underscore if not already present.
    """
    n = name.strip().replace(",", "").replace(" ", "_").lower()
    if not n.startswith("_"):
        n = "_" + n
    return n

def normalize_fk_name(name):
    """
    Normalizes a foreign key column name.
    - Strips extra spaces.
    - Replaces spaces with underscores.
    - Converts to lowercase.
    """
    return name.strip().lower().replace(" ", "_")

# Determine the base directory where your database folders reside.
# This script is located in: enas/tabert/wikidbs-public/scripts/
# And your databases are in: enas/tabert/wikidbs-10k/databases/
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../wikidbs-10k/databases")

# Lists to store rows for our Excel sheets
db_table_rows = []  # For "Databases_Tables" sheet: Database and Table names.
fk_rows = []        # For "ForeignKeys" sheet: Database, Source Table, Destination Table, and Foreign Key.

# Iterate over every subdirectory in base_dir (each is considered a database folder)
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue  # Skip if not a folder

    # Look for a JSON file in the database folder:
    # Prefer info_full.json; if it doesn't exist, try info_short.json.
    candidate_full = os.path.join(folder_path, "info_full.json")
    candidate_short = os.path.join(folder_path, "info_short.json")
    if os.path.exists(candidate_full):
        json_file_path = candidate_full
    elif os.path.exists(candidate_short):
        json_file_path = candidate_short
    else:
        print(f"Skipping folder {folder} (no JSON file found).")
        continue

    # Open and parse the JSON file.
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get the database name from the JSON INFO field; if missing, fall back on the folder name.
    db_name = data.get("INFO", {}).get("db_folder_name", folder)
    
    # Get the table definitions.
    tables = data.get("TABLES", {})
    for table_name, table_info in tables.items():
        # For a consistent view, normalize the table name.
        norm_table_name = normalize_table_name(table_name)
        db_table_rows.append({
            "Database": db_name,
            "Table": norm_table_name
        })
        
        # Check for foreign key relationships in this table.
        for fk_def in table_info.get("FOREIGN_KEYS", []):
            fk_list = fk_def.get("FOREIGN_KEY", [])
            # We expect that the second element is the referencing column name.
            if len(fk_list) < 2:
                continue
            fk_column_raw = fk_list[1]
            norm_fk_col = normalize_fk_name(fk_column_raw)
            
            # The destination table is provided under "REFERENCE_TABLE".
            ref_table = fk_def.get("REFERENCE_TABLE", "").strip()
            # Normalize destination table name.
            norm_ref_table = normalize_table_name(ref_table)
            
            fk_rows.append({
                "Database": db_name,
                "Source Table": norm_table_name,
                "Destination Table": norm_ref_table,
                "Foreign Key": norm_fk_col
            })

# Create pandas DataFrames for the two sheets.
df_db_tables = pd.DataFrame(db_table_rows)
df_fk = pd.DataFrame(fk_rows)

# Write the DataFrames to an Excel file with two sheets.
output_file = "database_summary.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_db_tables.to_excel(writer, sheet_name="Databases_Tables", index=False)
    df_fk.to_excel(writer, sheet_name="ForeignKeys", index=False)

print(f"Excel file '{output_file}' created successfully!")
