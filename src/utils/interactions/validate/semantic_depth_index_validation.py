import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_data(
    semantic_depth_path: str,
    validation_sample_path: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    with open(semantic_depth_path, "r", encoding="utf-8") as f:
        semantic_depth_data = json.load(f)
    
    with open(validation_sample_path, "r", encoding="utf-8") as f:
        validation_sample_data = json.load(f)
    
    #Filter users that appears in validation_sample_data
    semanthic_depth_filtered = {}
    for user_id, data in semantic_depth_data.items():
        if user_id in list(validation_sample_data.keys()):
            semanthic_depth_filtered[user_id] = data
            
    return semanthic_depth_filtered, validation_sample_data

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

def compute_metrics(
    semanthic_depth_filtered: dict[str, dict],
    validation_sample: dict[str, dict],
    output_dir: str
) -> dict[str, float]:
    # Get vectors of semantic depth levels and human validation for each user
    semantic_depth_vector = get_semantic_depth_vector(semanthic_depth_filtered)
    validation_vector = get_semantic_depth_vector(validation_sample)

    # Generate confusion matrix
    cm = confusion_matrix(validation_vector, semantic_depth_vector)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')  # Save the figure

    