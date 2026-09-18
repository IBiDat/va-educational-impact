###########################################################################################

# --- IMPORTS ---

import os, sys, json, logging

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths (using __file__ for reliability instead of getcwd)
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..', '..')
sys.path.append(project_path)

# Data directories
semantic_depth_filename = 'semantic_depth_data.json'
validation_sample = 'semantic_depth_data_validated.json'
semantic_depth_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', semantic_depth_filename)
validation_sample_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', validation_sample)
interaction_type_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'interactions_type', 'interaction_type.parquet')
validation_interaction_type_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'interaction_type_data_validated.json')
validations_cheating_interactions_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'cheating_interactions_validated.json')

output_dir = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'plots')

os.makedirs(output_dir, exist_ok=True)
###########################################################################################

from src.utils.interactions.validate.llm_answers_validation_utils import (
    load_data,
    compute_metrics
)

def main():
    #1. Load Necessary Data
    semanthic_depth_filtered, validation_sample, interactions_type_filtered, validation_interaction_type_data = load_data(
        semantic_depth_path,
        validation_sample_path,
        interaction_type_path,
        validation_interaction_type_path
    )
    
    #2. Compute metrics
    #WSDI
    compute_metrics(
        semanthic_depth_filtered,
        validation_sample, 
        value="semantic_depth_level",
        output_path=os.path.join(output_dir, 'confusion_matrix_wsdi.png')
    )
    
    #Interaction type
    compute_metrics(
        interactions_type_filtered,
        validation_interaction_type_data,
        value="interaction_type", 
        output_path=os.path.join(output_dir, 'confusion_matrix_interaction_type.png')
    )
    
    #Cheating interactions
    with open(validations_cheating_interactions_path, "r", encoding="utf-8") as f:
            validation_cheating_interactions: dict[str, dict[str|bool]] = json.load(f)
    cheating_interacitons_filtered = {
        k: {"cheating": True}
        for k in validation_cheating_interactions.keys()
    }
    
    compute_metrics(
        cheating_interacitons_filtered,
        validation_cheating_interactions,
        value="cheating", 
        output_path=os.path.join(output_dir, 'confusion_matrix_cheating_interactions.png')
    )

if __name__ == "__main__":
    main()