import polars as pl

def segment_groups(forms_interactions_df):

    for version in ['v1', 'v2', 'v3']:
        forms_interactions_df = forms_interactions_df.with_columns(
            pl.when(pl.col('grupo') == 'experimental')
            .then(pl.col(f'experimental_type_freq_quality_{version}'))
            .otherwise(pl.col('grupo'))
            .alias(f'grupo_segmented_{version}')
        )

    return forms_interactions_df

def compute_time_interval_variables(
    combined_df: pl.DataFrame
) -> pl.DataFrame:
    #Select relevant variables for analysis
    filtered_df = combined_df.select(
        #Identificadores Usuario
        "id",
        "grupo",
        "puntuacion_tc_hake_gain",
        
        #Variables Pre
        "marca temporal_pre",
        "puntuacion_tc_pre",
        
        #Variables Post
        "marca temporal_post",
        "puntuacion_tc_post",
        
        #Evaluation Timestamp
        "interactions_last_timestamp"
    )

    #Include Ganancia Unidades de Conocimiento
    filtered_df = filtered_df.with_columns(
        ganancia_conocimiento = (pl.col("puntuacion_tc_post") - pl.col("puntuacion_tc_pre"))*10
    )

    #Transform final_timestamp_interaction time
    filtered_df = filtered_df.with_columns(
        pl.col("interactions_last_timestamp")
        .dt.replace_time_zone("UTC")        # label it as UTC first (since it's naive)
        .dt.convert_time_zone("Europe/Madrid")  # converts respecting DST
        .dt.replace_time_zone(None)
        .alias("interactions_last_timestamp")
    )

    #Create groups (Instituto-Grupo)
    filtered_df = filtered_df.with_columns(
        class_group = (pl.col("id").str.split("-").list.get(0) + "_" + pl.col("grupo"))
    )
    
    #Transform temporal columns from str to datetime
    filtered_df = filtered_df.with_columns([
        pl.col("marca temporal_pre").str.to_datetime("%d/%m/%Y %H:%M:%S"),
        pl.col("marca temporal_post").str.to_datetime("%d/%m/%Y %H:%M:%S")
    ]).with_columns(
        pre_post_time_interval = (
            (pl.col("marca temporal_post") - pl.col("marca temporal_pre"))
            .dt.total_seconds()/60
        )
    )
    
    #Join DFs
    combined_df = combined_df.join(
        filtered_df.select(['id', 'pre_post_time_interval']),
        on="id",
        how="inner"
    )

    return combined_df