import os
import json

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_data(
    semantic_depth_path: str,
    validation_sample_path: str,
    interaction_type_path: str,
    validation_interaction_type_path: str
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    
    #WSDI
    with open(semantic_depth_path, "r", encoding="utf-8") as f:
        semantic_depth_data = json.load(f)
    
    with open(validation_sample_path, "r", encoding="utf-8") as f:
        validation_sample_data = json.load(f)
    
    #Filter users that appears in validation_sample_data
    semanthic_depth_filtered = {}
    for user_id, data in semantic_depth_data.items():
        if user_id in list(validation_sample_data.keys()):
            semanthic_depth_filtered[user_id] = data
    
    #Interaction type
    interactions_type_df = pl.read_parquet(interaction_type_path)
        
    with open(validation_interaction_type_path, "r", encoding="utf-8") as f:
        validation_interaction_type_data: dict[str, dict] = json.load(f)
    
    #Transform interactions_type_df into dict and filter by conversation_id that are in validation_interaction_type_data
    interaction_type_sample = {}
    for row in interactions_type_df.iter_rows(named=True):
        interaction_type_sample[row["conversation_id"]] = {
            "user_input": row["user_input"],
            "interaction_type": row["prompt_type"]
        }
    
    #Filter validation_interaction_type_data conversation_id
    interactions_type_filtered = {
        c: interaction_type_sample[c]
        for c in validation_interaction_type_data.keys()
    }
 
    return semanthic_depth_filtered, validation_sample_data, interactions_type_filtered, validation_interaction_type_data

def get_semantic_depth_vector(
    semantic_depth_dict: dict[str, dict]
) -> list[int]:
    # Extract the semantic depth levels for each user and return as a vector
    semantic_depth_vector = []
    for data in semantic_depth_dict.values():
        for response in data:
            # Assuming 'semantic_depth_level' is the key where the level is stored
            level = response.get('semantic_depth_level')
            semantic_depth_vector.append(level)
    
    return semantic_depth_vector

def dict_to_vector(
    entry_dict: dict[str, dict],
    value: str
) -> list[int]:
    #Extract the value for each user/conversation and return a vector
    vector = []
    if value == "semantic_depth_level":
        for data in entry_dict.values():
            for response in data:
                level = response.get(value)
                vector.append(level)
    else:
        for response in entry_dict.values():
            vector.append(response.get(value))
        
    return vector

def compute_metrics(
    llm_filtered: dict[str, dict],
    validation_sample: dict[str, dict],
    value: str,
    output_path: str
) -> dict[str, float]:
    # Get vectors of semantic depth levels and human validation for each user
    llm_vector = dict_to_vector(llm_filtered, value)
    validation_vector = dict_to_vector(validation_sample, value)

    # Generate confusion matrix
    cm = confusion_matrix(validation_vector, llm_vector)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(output_path, bbox_inches='tight')  # Save the figure

    # Generate classification report
    report = classification_report(validation_vector, llm_vector)
    print(f"\n\n{value}")
    print(report)