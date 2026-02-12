###########################################################################################
# To run this script successfully the VA Dockers have to be deployed
###########################################################################################

# --- IMPORTS ---

import os, sys, json, logging
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
interactions_raw_data_dir = os.path.join(project_path, 'data', 'interactions', 'raw_data')

# Output Files
# Ensure output directory exists
os.makedirs(interactions_raw_data_dir, exist_ok=True)
output_file_path = os.path.join(interactions_raw_data_dir, 'interactions_raw_data.json')

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
        interactions_col = db['interactions']
        hashes_col = db['access_hashes']
        logging.info(f" -> Connected successfully to DB: {DB_NAME}")

    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    # 2. Extract Data
    logging.info("STEP 2: Querying users and processing interactions...\n")
    
    interactions_data = {}
    
    try:
        # Step A: Get users with 'used': True
        query_users = {"used": True}
        count_users = hashes_col.count_documents(query_users)
        users_cursor = hashes_col.find(query_users)
        
        logging.info(f" -> Processing {count_users} users with 'used': True ...")

        for user_doc in users_cursor:
            user_id = user_doc.get('user_id')
            participant_hash = user_doc.get('hash') 
            
            if not user_id or not participant_hash:
                continue 

            # Initialize structure
            interactions_data[participant_hash] = {
                'chat_interactions': [],
                'evaluation_interactions': {} 
            }

            # Step B: Get interactions per user
            user_interactions = interactions_col.find({'user_id': user_id}).sort('timestamp', 1)

            for interaction in user_interactions:
                i_type = interaction.get('type')

                if i_type == 'chat_interaction':
                    chat_entry = {
                        'user_input': interaction.get('user_input', ''),
                        'model_response': interaction.get('model_response', '')
                    }
                    interactions_data[participant_hash]['chat_interactions'].append(chat_entry)

                elif i_type == 'evaluation':
                    eval_entry = {
                        'evaluation_questions': interaction.get('evaluation_questions', ''),
                        'user-answers': interaction.get('user_answers', '')
                    }
                    interactions_data[participant_hash]['evaluation_interactions'] = eval_entry
        
        logging.info(f" -> Extracted data for {len(interactions_data)} participants.\n")

    except Exception as e:
        logging.error(f"Error during data extraction: {e}")
        return

    # 3. Save Outputs
    logging.info("STEP 3: Saving results to JSON...\n")

    if os.path.exists(output_file_path):
        logging.warning("⚠️  ALERTA:")
        logging.warning(f"   The file '{os.path.basename(output_file_path)}' ALREADY EXISTS.")
        logging.warning("   ❌ Operation cancelled: File was NOT overwritten.")
        logging.warning("   (Please delete the old file or rename the output configuration).")
        sys.exit(1) # Exit cleanly but indicating no save occurred
    else:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(interactions_data, f, ensure_ascii=False, indent=4)
            
            logging.info(f" -> Saved interactions data: {output_file_path}")
            logging.info(f" -> Total exported users: {len(interactions_data.keys())}\n")

        except Exception as e:
            logging.error(f"Failed to save output files: {e}\n")
            sys.exit(1)

    logging.info("✅ EXTRACTION PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()