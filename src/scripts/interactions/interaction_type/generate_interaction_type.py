###########################################################################################

# --- IMPORTS ---

import os, sys
from dotenv import load_dotenv
load_dotenv()

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths (using __file__ for reliability instead of getcwd)
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..', '..')
sys.path.append(project_path)

from src.utils.interactions.interaction_type.interaction_type_generation import (
    return_interaction_type
)

# Data directories
template_path = os.path.join(project_path, 'data', 'interactions', 'templates', 'generate_interaction_type.md')
raw_data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'semantic_depth_data.json')
output_dir = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'interactions_type')

os.makedirs(output_dir, exist_ok=True)
###########################################################################################

def main():
    #1. Get interaction type using LLM
    interaction_type_df, grouped_interaction_type_df = return_interaction_type(
        raw_data_path=raw_data_path,
        template_path=template_path
    )
    
    #2. Save DFs
    interaction_type_df.write_parquet(
        os.path.join(
            output_dir,
            "interaction_type.parquet"
        )
    )
    
    grouped_interaction_type_df.write_parquet(
        os.path.join(
            output_dir,
            "id_grouped_prompt_type.parquet"
        )
    )
    

if __name__ == "__main__":
    main()