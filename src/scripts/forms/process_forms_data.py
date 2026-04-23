###########################################################################################

# --- IMPORTS ---

import os, sys, logging
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
data_dir = os.path.join(project_path, 'data', 'forms')
raw_data_dir = os.path.join(data_dir, 'raw_data')
processed_data_dir = os.path.join(data_dir, 'processed_data')
hashes_dir = os.path.join(project_path, 'data', 'hashes', 'processed_data')

###########################################################################################

# --- LOCAL IMPORTS ---

from src.utils.forms.forms_data_processing import process_forms_data

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow for Forms Data Processing.
    Processes pre/post forms, cleans IDs, crosses data, and adds group hashes.
    """
    logging.info("▶️ STARTING FORMS DATA PROCESSING PIPELINE")

    # 1. Load Data
    logging.info("STEP 1: Loading raw data files and hashes...\n")

    try:
        raw_data_filenames = [f for f in os.listdir(raw_data_dir) if 'forms' in f]
        raw_data = {f.split('.')[0]: pl.read_csv(os.path.join(raw_data_dir, f)) for f in raw_data_filenames}
        logging.info(f" -> Loaded {len(raw_data_filenames)} raw forms files")

        hashes_groups_df = pl.read_csv(os.path.join(hashes_dir, "hashes_groups.csv"), separator=";")
        logging.info(f" -> Loaded hashes groups file\n")

    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Process Forms Data
    logging.info("STEP 2: Processing forms data via util functions...\n")

    try:
        processed_forms_data = process_forms_data(raw_data_dir=raw_data_dir, processed_data_dir=processed_data_dir)
        df_pre = processed_forms_data['processed_respuestas_forms_pre.csv']
        df_post = processed_forms_data['processed_respuestas_forms_post.csv']
        logging.info(f" -> Forms processed. Pre size: {df_pre.shape[0]}, Post size: {df_post.shape[0]}\n")

    except Exception as e:
        logging.error(f"Error during forms processing: {e}")
        sys.exit(1)

    # 3. Clean and Fix Data
    logging.info("STEP 3: Cleaning data and fixing manual errors...\n")

    try:
        # Fix typo in student ID
        fix_expr = (
            pl.when(pl.col("id") == "RamiroMaeztu-rya")
            .then(pl.lit("RamiroMaeztu-kya"))
            .otherwise(pl.col("id"))
            .alias("id")
        )
        
        df_pre = df_pre.with_columns(fix_expr)
        df_post = df_post.with_columns(fix_expr)
        logging.info(" -> ID manual fixes applied successfully\n")

    except Exception as e:
        logging.error(f"Error fixing data: {e}")
        sys.exit(1)

    # 4. Cross Pre/Post Data
    logging.info("STEP 4: Renaming and crossing pre/post dataframes...\n")

    try:
        cols_protegidas = ['id', 'centro']
        df_pre_suf = df_pre.rename({col: f"{col}_pre" for col in df_pre.columns if col not in cols_protegidas})
        df_post_suf = df_post.rename({col: f"{col}_post" for col in df_post.columns if col not in cols_protegidas})

        df_cruzado = df_pre_suf.join(df_post_suf, on='id', how='inner')
        logging.info(f" -> Pre/Post inner join completed. Crossed records: {df_cruzado.shape[0]}\n")

    except Exception as e:
        logging.error(f"Error during cross join: {e}")
        sys.exit(1)

    # 5. Join Groups Data
    logging.info("STEP 5: Joining groups to crossed dataframe...\n")

    try:
        df_grupos = hashes_groups_df.select(["id", "grupo"])
        
        df_cruzado = df_cruzado.join(
            df_grupos,
            on="id",
            how="left"
        )
        logging.info(" -> Groups join completed successfully\n")

    except Exception as e:
        logging.error(f"Error during groups join: {e}")
        sys.exit(1)

    # 6. Save Outputs
    logging.info("STEP 6: Saving results to Parquet...\n")

    try:
        df_cruzado_filename = 'processed_pre_post_forms.parquet'
        filepath = os.path.join(processed_data_dir, df_cruzado_filename)
        
        df_cruzado.write_parquet(filepath)
        logging.info(f" -> Saved: {filepath}\n")

    except Exception as e:
        logging.error(f"Failed to save output files: {e}")
        sys.exit(1)

    logging.info("✅ PROCESSING PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()