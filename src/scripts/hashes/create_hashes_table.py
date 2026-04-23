###########################################################################################

# --- IMPORTS ---

import os, sys, json, glob, logging
import polars as pl

###########################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')  
sys.path.append(project_path)

# Input/Output Directories
hashes_dir = os.path.join(project_path, 'data', 'hashes', 'raw_data')
output_dir = os.path.join(project_path, 'data', 'hashes', 'processed_data')

# Output File
output_csv_path = os.path.join(output_dir, 'hashes_groups_old.csv')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

###########################################################################################

# --- IMPORTS (PROJECT MODULES) ---

from src.utils.hashes.hashes_table_creation import process_school_hashes

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main flow: Finds JSON files, assigns groups per school, and exports CSV.
    """
    logging.info("▶️ STARTING GROUP ASSIGNMENT PIPELINE")

    # 1. Find Files
    logging.info("STEP 1: Scanning for hash files...\n")
    
    # Busca todos los .json en la carpeta data/hashes
    json_files = glob.glob(os.path.join(hashes_dir, "*.json"))
    
    if not json_files:
        logging.error(f"No JSON files found in {hashes_dir}")
        sys.exit(1)
        
    logging.info(f" -> Found {len(json_files)} files to process.")

    # 2. Process Data
    logging.info("STEP 2: Processing schools and assigning groups...\n")

    dataframes = []

    for file_path in json_files:
        filename = os.path.basename(file_path)
        logging.info(f" -> Processing: {filename}...")
        
        df_school = process_school_hashes(file_path)
        
        if df_school is not None:
            dataframes.append(df_school)

    if not dataframes:
        logging.error("No data could be processed. Exiting.")
        sys.exit(1)

    # 3. Concatenate and Export
    logging.info("\nSTEP 3: Consolidating and saving CSV...\n")

    try:
        # Unir todos los dataframes verticales (stack)
        full_df = pl.concat(dataframes)

        # Guardar como CSV separado por punto y coma
        full_df.write_csv(output_csv_path, separator=";")

        logging.info(f" -> Total records processed: {full_df.height}")
        logging.info(f" -> Breakdown by Group:\n{full_df['grupo'].value_counts()}")
        logging.info(f"✅ CSV saved successfully at: {output_csv_path}")

    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
        sys.exit(1)

###########################################################################################

if __name__ == "__main__":
    main()