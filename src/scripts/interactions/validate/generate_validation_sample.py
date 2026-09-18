###########################################################################################

# --- IMPORTS ---

import os, sys, json, argparse

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths (using __file__ for reliability instead of getcwd)
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..', '..')
sys.path.append(project_path)

# Data directories
raw_data_filename = 'interactions_raw_data.json'
raw_data_path = os.path.join(project_path, 'data', 'interactions', 'raw_data', raw_data_filename)
interactions_processed_data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'interactions_processed_data.parquet')
interaction_type_data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'interactions_type', 'interaction_type.parquet')
cheating_interaction_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'cheating_score', 'interaction_cheating_score.parquet')

output_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'semantic_depth_data_validated.json')
output_path_interaction_type = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'interaction_type_data_validated.json')
output_path_cheating_interactions = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'cheating_interactions_validated.json')
###########################################################################################

# --- LOCAL IMPORTS ---

from src.utils.interactions.validate.validation_sample_generation import (
    load_raw_data,
    filter_users_for_validation_sample,
    filter_interactions_type_for_validation,
    filter_cheating_interactions_for_validation
)
# --- MAIN EXECUTION ---

def main():
    #1. Load Necessary Data
    raw_data, interactions_processed_data, interactions_type_data, cheating_interaction_data = load_raw_data(
        raw_data_path,
        interactions_processed_data_path,
        interaction_type_data_path, 
        cheating_interaction_path
    )
    
    #2. Filter users for WSDI
    filtered_raw_data = filter_users_for_validation_sample(
        raw_data,
        interactions_processed_data,
        sample_size=15
    )
    
    #3. Filter interactions for interactions_type validation
    filtered_interactions_type = filter_interactions_type_for_validation(
        interactions_type_data,
        sample_size=50
    )
    
    #4. Filter interactions for interactions_type validation
    filtered_cheating_interactions = filter_cheating_interactions_for_validation(
        cheating_interaction_data,
        sample_size=50
    )
    
    #5. Include null semantic_depth_level for evaluation
    sampled_users = {}
    for user_id, chat_interactions in filtered_raw_data.items():
        new_chat_interactions = []
        for interaction in chat_interactions:
            interaction['semantic_depth_level'] = None
            new_chat_interactions.append(interaction)
        
        sampled_users[user_id] = new_chat_interactions
    
    #6. Repeat the process for filtered_interactions_type
    interaction_type_sample = {}
    for row in filtered_interactions_type.iter_rows(named=True):
        interaction_type_sample[row["conversation_id"]] = {
            "user_input": row["user_input"],
            "interaction_type": None
        }
    
    #7. Repeat the process for filtered_interactions_type
    cheating_sample = {}
    for row in filtered_cheating_interactions.iter_rows(named=True):
        cheating_sample[row["conversation_id"]] = {
            "user_input": row["user_input"],
            "most_similar_question": row["most_similar_question"],
            "cheating": None
        }
    
    #8. Save the validation sample
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"WSDI validation file already exists. Check the path or remove the existing file before running the script.")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_users, f, ensure_ascii=False, indent=4)
    
    if os.path.exists(output_path_interaction_type):
            print(f"Interaction type validation file already exists. Check the path or remove the existing file before running the script.")
    else:
        with open(output_path_interaction_type, "w", encoding="utf-8") as f:
            json.dump(interaction_type_sample, f, ensure_ascii=False, indent=4)
    
    if os.path.exists(output_path_cheating_interactions):
        print(f"Cheating interactions validation file already exists. Check the path or remove the existing file before running the script.")
    else:
        with open(output_path_cheating_interactions, "w", encoding="utf-8") as f:
            json.dump(cheating_sample, f, ensure_ascii=False, indent=4)
    
###########################################################################################

if __name__ == "__main__":
    main()