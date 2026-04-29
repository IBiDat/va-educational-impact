###########################################################################################

# --- IMPORTS ---

import os, sys, json, logging
import polars as pl

###########################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths (using __file__ for reliability instead of getcwd)
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
sys.path.append(project_path)

# Data directories
raw_data_filename = 'interactions_raw_data_20260422_112415.json'
semantic_depth_data_filename = 'semantic_depth_data_20260422_203030.json'
raw_data_path = os.path.join(project_path, 'data', 'interactions', 'raw_data', raw_data_filename)
semantic_depth_data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', semantic_depth_data_filename)
output_dir = os.path.join(project_path, 'data', 'interactions', 'processed_data')

###########################################################################################

# --- LOCAL IMPORTS ---

from src.utils.interactions.interactions_data_processing import (
    process_interactions_data,
    process_semantic_depth_data,
    process_combined_interactions_data,
    segment_experimental_type
)

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow for Interactions Data Processing.
    Computes the Weighted Semantic Depth Index (WSDI) and saves processed data as Parquet.
    """
    logging.info("▶️ STARTING INTERACTIONS DATA PROCESSING PIPELINE")

    # 1. Load Data
    logging.info("STEP 1: Loading data files...\n")

    try:
        with open(raw_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        logging.info(f" -> Loaded file: {raw_data_filename}")

        with open(semantic_depth_data_path, "r", encoding="utf-8") as f:
            semantic_depth_data = json.load(f)
        logging.info(f" -> Loaded file: {semantic_depth_data_filename}\n")

    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Process Interactions
    logging.info("STEP 2: Processing interactions and computing WSDI...\n")

    try:
        interactions_df = process_interactions_data(raw_data)
        logging.info(f" -> Interactions processed: {interactions_df.shape[0]} records")

        semantic_depth_df, weighted_semantic_depth_df = process_semantic_depth_data(semantic_depth_data)
        logging.info(f" -> WSDI computed: {weighted_semantic_depth_df.shape[0]} participants\n")

    except Exception as e:
        logging.error(f"Error during data processing: {e}")
        sys.exit(1)

    # 3. Join WSDI to Interactions
    logging.info("STEP 3: Joining WSDI to interactions dataframe...\n")

    try:
        interactions_df = interactions_df.join(
            weighted_semantic_depth_df[['id', 'WSDI', 'WSDI_cat']],
            how='left',
            on='id'
        )
        logging.info(f" -> Join completed successfully\n")

    except Exception as e:
        logging.error(f"Error during dataframe join: {e}")
        sys.exit(1)


    # 4. Process Combined Interactions Data
    logging.info("STEP 4: Processing combined interactions data...\n")

    interactions_df = process_combined_interactions_data(interactions_df)
    interactions_df = segment_experimental_type(interactions_df)

    # 5. Save Outputs
    logging.info("STEP 5: Saving results to Parquet...\n")

    output_files = {
        'semantic_depth.parquet': semantic_depth_df,
        'weighted_semantic_depth.parquet': weighted_semantic_depth_df,
        'interactions_processed_data.parquet': interactions_df,
    }

    try:
        for filename, df in output_files.items():
            filepath = os.path.join(output_dir, filename)
            df.write_parquet(filepath)
            logging.info(f" -> Saved: {filepath}")

        logging.info(f"\n -> Total output files saved: {len(output_files)}\n")

    except Exception as e:
        logging.error(f"Failed to save output files: {e}")
        sys.exit(1)

    logging.info("✅ PROCESSING PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()