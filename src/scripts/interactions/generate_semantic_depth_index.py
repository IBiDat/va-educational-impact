###########################################################################################

# --- IMPORTS ---

import os, sys, json, logging
from google import genai
from dotenv import load_dotenv
load_dotenv()

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
raw_data_filename = 'interactions_raw_data.json'
raw_data_path = os.path.join(project_path, 'data', 'interactions', 'raw_data', raw_data_filename)
output_dir = os.path.join(project_path, 'data', 'interactions', 'processed_data')

###########################################################################################

# --- LOCAL IMPORTS ---

from src.utils.interactions.interactions_data_processing import generate_semantic_depth_index

###########################################################################################

# -- CONFIG PARAMETERS --

MODEL = 'gemini-3-flash-preview'
TEMPERATURE = 0

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow for Semantic Depth Index Generation.
    Classifies each student interaction using an LLM and saves the results as JSON.
    """
    logging.info("▶️ STARTING SEMANTIC DEPTH INDEX GENERATION PIPELINE")

    # --- Configuration and Path Setup ---
    output_filename = 'semantic_depth_data.json'
    output_path = os.path.join(output_dir, output_filename)

    # Check for existing output to prevent accidental overwrite
    if os.path.exists(output_path):
        logging.warning(f"Aborting process: Output file '{output_filename}' already exists.")
        logging.info("Suggestion: Rename the output or delete the existing file before retrying.")
        sys.exit(0)  # Using exit code 0 if this is an expected safety guard

    # --- Step 1: Raw Data Ingestion ---
    logging.info(f"Initiating data load from: {raw_data_filename}")

    try:
        with open(raw_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        logging.info("Data successfully loaded into memory.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Critical error loading raw data: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during ingestion: {e}")
        sys.exit(1)
    
    # 2. Generate Semantic Depth Index
    logging.info("STEP 2: Generating semantic depth index via LLM...\n")

    try:
        client = genai.Client()

        semantic_depth_data = generate_semantic_depth_index(
            client = client,
            model = MODEL,
            temperature = TEMPERATURE,
            raw_data = raw_data
        )
        logging.info(f" -> Semantic depth index generated for {len(semantic_depth_data)} participants\n")

    except Exception as e:
        logging.error(f"Error during semantic depth index generation: {e}")
        sys.exit(1)

    # 3. Save Output
    logging.info("STEP 3: Saving semantic depth data to JSON...\n")

    try:

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(semantic_depth_data, f, ensure_ascii=False, indent=4)

            logging.info(f" -> Saved: {output_path}\n")

    except Exception as e:
        logging.error(f"Failed to save semantic depth data: {e}")
        sys.exit(1)

    logging.info("✅ SEMANTIC DEPTH INDEX GENERATION PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()