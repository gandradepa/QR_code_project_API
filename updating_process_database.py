import os
import pandas as pd
from datetime import datetime
import sqlite3

# =============================================================================
#  SECTION 1: FUNCTIONS FOR PROCESSING JPG PHOTOS AND UPDATING DATABASE
#  (This section is unchanged)
# =============================================================================

def process_jpg_files(directory_path):
    """
    Reads all JPG files in a directory, extracts a 10-character ID and the
    file creation date, and returns the data in a pandas DataFrame.
    """
    print("--- Starting JPG file processing... ---")
    file_ids = []
    creation_dates = []
    df = pd.DataFrame()

    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                base_name = os.path.splitext(filename)[0]
                qr_code_id = base_name[:10]
                file_ids.append(qr_code_id)
                
                full_file_path = os.path.join(directory_path, filename)
                creation_timestamp = os.path.getctime(full_file_path)
                creation_datetime = datetime.fromtimestamp(creation_timestamp)
                creation_dates.append(creation_datetime)

        data = {'QR_code_ID': file_ids, 'date_set': creation_dates}
        df = pd.DataFrame(data)
        print("✅ DataFrame created successfully from photos.")
        print(df.head())
    else:
        print(f"❌ Error: The photo directory '{directory_path}' was not found.")
    
    return df

def update_qr_codes_table(df, db_path):
    """
    Updates the 'date_set' column in the 'QR_codes' table of an SQLite
    database using data from the provided DataFrame.
    """
    print("\n--- Starting database update for QR_codes table... ---")
    df['date_set'] = df['date_set'].astype(str)
    
    connection = None
    try:
        print(f"🔄 Connecting to database at '{db_path}'...")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        sql_update_query = "UPDATE QR_codes SET date_set = ? WHERE QR_code_ID = ?"

        print("🚀 Starting database update process...")
        for index, row in df.iterrows():
            cursor.execute(sql_update_query, (row['date_set'], row['QR_code_ID']))
        
        connection.commit()
        print(f"\n✅ Success! Updated {len(df)} records in the database.")

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        print("No changes were saved. The transaction has been rolled back.")

    finally:
        if connection:
            connection.close()
            print("🔌 Database connection closed.")

# =============================================================================
#  SECTION 2: FUNCTIONS FOR PROCESSING JSON FILES AND SAVING TO DATABASE
#  (This section has been updated)
# =============================================================================

def process_json_files(path, db_path):
    """
    Identifies JSON files, finds the unique codes, and fetches the creation
    date for each code from the 'QR_codes' database table.
    """
    if not os.path.exists(path):
        print(f"--- ERROR ---")
        print(f"The specified path does not exist: {path}")
        return pd.DataFrame(columns=["code", "create_date"])

    # --- Step 1: Scan JSON files and deduplicate to get a unique list of codes ---
    data_list = []
    print("--- Starting JSON file processing... ---")
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            try:
                code = filename[:10]
                timestamp = os.path.getmtime(file_path) # Use mod time for deduplication only
                data_list.append({"code": code, "mod_date": datetime.fromtimestamp(timestamp)})
            except Exception as e:
                print(f"An unexpected error occurred with file '{filename}': {e}")
    
    if not data_list:
        print("--- No JSON files were found or processed. ---")
        return pd.DataFrame(columns=["code", "create_date"])

    temp_df = pd.DataFrame(data_list)
    temp_df = temp_df.sort_values(by='mod_date', ascending=False)
    unique_codes_df = temp_df.drop_duplicates(subset=['code'], keep='first')[['code']].reset_index(drop=True)
    print(f"Found {len(unique_codes_df)} unique codes to process from JSON files.")

    # --- Step 2: Fetch creation dates from the database ---
    print(f"\n--- Fetching creation dates from database: {db_path} ---")
    final_df = pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        # Read the relevant columns from the QR_codes table
        qr_data_df = pd.read_sql_query("SELECT QR_code_ID, date_set FROM QR_codes", conn)
        conn.close()
        
        # Merge the unique codes from JSON files with the data from the database
        merged_df = pd.merge(
            unique_codes_df,
            qr_data_df,
            left_on='code',
            right_on='QR_code_ID',
            how='left'  # Keep all codes from JSON files, even if no DB match
        )
        
        # Prepare the final DataFrame with the correct column names
        final_df = merged_df[['code', 'date_set']].copy()
        final_df.rename(columns={'date_set': 'create_date'}, inplace=True)
        print("✅ Successfully merged file codes with database creation dates.")

    except sqlite3.Error as e:
        print(f"❌ Database error while fetching creation dates: {e}")
        return pd.DataFrame(columns=["code", "create_date"])

    return final_df

def save_json_to_sqlite(df, db_path, table_name):
    """Saves a DataFrame to a new SQLite database table, replacing it if it exists."""
    print(f"\n--- Connecting to database: {db_path} ---")
    try:
        conn = sqlite3.connect(db_path)
        print("Connection successful.")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Successfully saved {len(df)} rows to table '{table_name}'.")
    except sqlite3.Error as e:
        print(f"--- Database Error: {e} ---")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")

# =============================================================================
#  SECTION 3: MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Define paths for both processes
    jpg_directory_path = r"/home/developer/Capture_photos_upload"
    json_directory_path = r"/home/developer/Output_jason_api"
    database_path = r"/home/developer/asset_capture_app_dev/data/QR_codes.db"
    
    # --- Task 1: Process JPGs and Update the 'QR_codes' Table ---
    jpg_df = process_jpg_files(jpg_directory_path)
    if not jpg_df.empty:
        update_qr_codes_table(jpg_df, database_path)

    print("\n" + "="*50 + "\n")

    # --- Task 2: Process JSONs and Create/Replace the 'json_files' Table ---
    # UPDATED: Pass the database_path to the function so it can fetch dates
    json_df = process_json_files(json_directory_path, database_path)
    if not json_df.empty:
        print("\n--- Resulting DataFrame (with dates from database) ---")
        print(json_df)
        print("\n--- DataFrame Info ---")
        json_df.info()
        save_json_to_sqlite(json_df, database_path, "json_files")