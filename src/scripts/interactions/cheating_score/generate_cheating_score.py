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

from src.utils.interactions.cheating_score.cheating_score_generation import (
    get_forms_questions,
    get_static_content_questions,
    compute_cheating_score
)

# Data directories
raw_data_filename = 'interactions_raw_data.json'
raw_data_path = os.path.join(project_path, 'data', 'interactions', 'raw_data', raw_data_filename)
template_path = os.path.join(project_path, 'data', 'interactions', 'templates', 'question_copy_detection_prompt.md')
forms_dir = os.path.join(project_path, 'data', 'forms', 'raw_data')
static_content_path = os.path.join(project_path, 'data', 'static_content', 'raw_data', 'static_content_raw_data.json')
output_dir = os.path.join(project_path, 'data', 'interactions', 'processed_data', 'cheating_score')

os.makedirs(output_dir, exist_ok=True)
###########################################################################################

def main():
    #1. Extract forms questions
    forms_questions = get_forms_questions(
        forms_dir=forms_dir
    )
    
    #2. Extract static content questions
    static_content_questions = get_static_content_questions(
        static_content_path=static_content_path
    )
    
    #3. Concat all the questions in an unique list
    every_question = forms_questions + static_content_questions
    
    #4. Compute similarity score and compute the cheating score for each conversation/user
    cheating_df, grouped_cheating_df = compute_cheating_score(
        raw_data_path=raw_data_path,
        template_path=template_path,
        every_question=every_question
    )
    
    #5. Save DFs
    cheating_df.write_parquet(
        os.path.join(
            output_dir,
            "interaction_cheating_score.parquet"
        )
    )
    
    grouped_cheating_df.write_parquet(
        os.path.join(
            output_dir,
            "id_cheating_score.parquet"
        )
    )
    

if __name__ == "__main__":
    main()