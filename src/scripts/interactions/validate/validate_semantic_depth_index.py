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
semantic_depth_filename = 'semantic_depth_data_ORIGINAL.json'
validation_sample = 'semantic_depth_data_validated.json'
semantic_depth_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', semantic_depth_filename)
validation_sample_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', validation_sample)
output_dir = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'plots')

os.makedirs(output_dir, exist_ok=True)
###########################################################################################

from src.utils.interactions.validate.semantic_depth_index_validation import (
    load_data,
    compute_metrics
)

def main():
    #1. Load Necessary Data
    semanthic_depth_filtered, validation_sample = load_data(
        semantic_depth_path,
        validation_sample_path
    )
    
    #2. Compute metrics
    compute_metrics(
        semanthic_depth_filtered,
        validation_sample, 
        output_dir
    )

if __name__ == "__main__":
    main()