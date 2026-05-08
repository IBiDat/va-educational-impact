###########################################################################################
# WARNING: To run this script successfully the VA Dockers have to be deployed
###########################################################################################

# --- IMPORTS ---

import os, sys, json, logging
from datetime import datetime
from pymongo import MongoClient

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
static_content_raw_data_dir = os.path.join(project_path, 'data', 'static_content', 'raw_data')

# Output Files
# Ensure output directory exists
os.makedirs(static_content_raw_data_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_file_path = os.path.join(static_content_raw_data_dir, f'static_content_raw_data.json')

# Database Configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "universia_VA"

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow for Interactions Data Extraction.
    Extracts user interactions from MongoDB and saves them as JSON.
    """
    logging.info("▶️ STARTING INTERACTIONS DATA EXTRACTION PIPELINE")

    # 1. Connection to Database
    logging.info("STEP 1: Connecting to MongoDB...\n")
    
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        reviewed_col = db['reviewed_col']
        vid_metadata_col = db['vid_metadata_col']
        logging.info(f" -> Connected successfully to DB: {DB_NAME}")

    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    # 2. Extract Data
    logging.info("STEP 2: Querying users and processing interactions...\n")
    
    try:    
        static_content = {}   
        for collection in ['vid_metadata_col', 'reviewed_col']:
            static_content[collection] = {}
            for doc in db[collection].find():
                for data_name, data in doc.items():
                    if data_name != '_id':
                        static_content[collection][data_name] = data       
       
        logging.info(f" -> Extracted static content.\n")

    except Exception as e:
        logging.error(f"Error during static content extraction: {e}")
        return

    # 3. Save Outputs
    logging.info("STEP 3: Saving results to JSON...\n")

    if os.path.exists(json_file_path):
        logging.warning("⚠️  ALERTA:")
        logging.warning(f"   The file '{os.path.basename(json_file_path)}' ALREADY EXISTS.")
        logging.warning("   ❌ Operation cancelled: File was NOT overwritten.")
        logging.warning("   (Please delete the old file or rename the output configuration).")
        sys.exit(1) # Exit cleanly but indicating no save occurred
    else:
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(static_content, f, ensure_ascii=False, indent=4)
            
            logging.info(f" -> Saved static content data: {json_file_path}")

        except Exception as e:
            logging.error(f"Failed to save output files: {e}\n")
            sys.exit(1)

    logging.info("✅ EXTRACTION PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()