import os, sys, json
import polars as pl

def load_raw_data(
    raw_data_path: str,
    interactions_processed_data_path: str,
    interaction_type_data_path: str
) -> tuple[dict[str, dict], pl.DataFrame, pl.DataFrame]:
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    interactions_processed_data = pl.read_parquet(interactions_processed_data_path)
    
    interactions_type_data = pl.read_parquet(interaction_type_data_path)
    
    return raw_data, interactions_processed_data, interactions_type_data

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

def filter_interactions_type_for_validation(
    interactions_type_data: pl.DataFrame,
    sample_size: int
) -> pl.DataFrame:
    #Filter conversations with TRANSFERENCE type of interaction
    transfer = interactions_type_data.filter(
        pl.col("prompt_type") == "TRANSFERENCE"
    )
    
    #Sample the remaining conversations until reaching sample_size
    rest = interactions_type_data.filter(
        pl.col("prompt_type") != "TRANSFERENCE"
    ).sample(
        sample_size - transfer.height,
        seed=33
    )
    
    #Concat and suffle
    filtered_interactions_type: pl.DataFrame = pl.concat(
        [
            transfer,
            rest
        ]
    ).sample(
        fraction=1.0,
        shuffle=True,
        seed=33
    )
    
    return filtered_interactions_type