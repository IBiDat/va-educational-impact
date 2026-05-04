import os, sys, json
import polars as pl

def load_raw_data(
    raw_data_path: str,
    interactions_processed_data_path: str
) -> tuple[dict[str, dict], pl.DataFrame]:
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    interactions_processed_data = pl.read_parquet(interactions_processed_data_path)
    
    return raw_data, interactions_processed_data

def filter_users_for_validation_sample(
    raw_data: dict[str, dict],
    interactions_processed_data: pl.DataFrame,
    sample_size: int
) -> dict[str, dict]:
    # Filter users whose chat_interactions_counts is 0
    filtered_processed_data = interactions_processed_data.filter(
        pl.col("chat_interactions_counts") > 0
    ).sample(sample_size, seed=42) # Sample users for validation, ensuring they have interactions to review
    
    #Filter out these users from raw data
    filtered_raw_data = {
        user_id: data["chat_interactions"] for user_id, data in raw_data.items() 
        if user_id in filtered_processed_data['id'].to_list()
    }
    
    
    return filtered_raw_data