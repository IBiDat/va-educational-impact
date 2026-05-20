#########################################################################################################################################################

import os
import logging
import json
import re
import polars as pl
from google.genai import types
from typing import TypedDict

import statistics
import numpy as np
from datetime import datetime

#########################################################################################################################################################

file_path = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(file_path, '..', '..', '..')
template_path = os.path.join(directory_path, 'data', 'interactions', 'templates', 'semantic_depth_evaluation_template.md')
summary_path = os.path.join(directory_path, 'data', 'static_content', 'processed_data', 'summary_video_transcription.txt')
transcription_path = os.path.join(directory_path, 'data', 'static_content', 'raw_data', 'static_content_raw_data.json')
validation_sample_path = os.path.join(directory_path, 'data', 'interactions', 'processed_data', 'validation_samples', 'semantic_depth_data_validated.json')

#########################################################################################################################################################

class SemanticDepthOutput(TypedDict):
    reasoning_semantic_depth_level: str
    semantic_depth_level: int

#########################################################################################################################################################

def process_interactions_data(raw_data):

    def _get_timestamps(data_id: dict) -> tuple[datetime | None, list[datetime]| None]:
        """Extrae el timestamp de evaluación y todos los timestamps del chat para un id."""
        eval_ts = (
            datetime.fromisoformat(data_id['evaluation_interactions']['timestamp'])
            if data_id['evaluation_interactions']
            else None
        )
        chat_ts = (
            [
                datetime.fromisoformat(i["timestamp"])
                for i in data_id['chat_interactions']
            ]
            if data_id['chat_interactions']
            else None
        )
        
        return eval_ts, chat_ts
    
    def calculate_gini_chat_interaction_distribution_score(
        timestamps: list[datetime]|None
    ):
        """
        Calculates a score from 0.0 to 1.0 indicating how evenly distributed 
        a user's interactions are across their own timeline.
        
        Args:
            timestamps (list of datetime): List of interaction datetimes.
            
        Returns:
            float: Score between 0.0 (highly concentrated) and 1.0 (highly distributed).
        """
        # If there are fewer than 3 interactions, distribution cannot be meaningfully measured
        if not timestamps or len(timestamps) < 3:
            return 0.0
            
        # 1. Sort timestamps chronologically
        timestamps = sorted(timestamps)
        
        # 2. Calculate time differences (in seconds) between consecutive interactions
        timestamps_seconds = [t.timestamp() for t in timestamps]
        time_differences = np.diff(timestamps_seconds)
        
        # If all interactions occurred at the exact same second
        if np.sum(time_differences) == 0:
            return 0.0
            
        # 3. Sort differences in ascending order (required for Gini calculation)
        sorted_differences = np.sort(time_differences)
        n = len(sorted_differences)
        
        # 4. Calculate the raw Gini Coefficient
        indices = np.arange(1, n + 1)
        numerator = np.sum((2 * indices - n - 1) * sorted_differences)
        denominator = n * np.sum(sorted_differences)
        
        gini_raw = numerator / denominator
        
        # 5. Apply sample size correction so that the maximum Gini is always 1.0
        if n > 1:
            corrected_gini = gini_raw * (n / (n - 1))
        else:
            corrected_gini = gini_raw
        
        # Cap at 1.0 to handle potential floating-point precision errors
        corrected_gini = min(corrected_gini, 1.0)
        
        # 6. Invert the Gini index so 1 means "highly distributed" and 0 means "highly concentrated"
        score = 1.0 - corrected_gini
        
        return score
    
    def process_interactions_timestamps(
        eval_ts: datetime|None,
        chat_ts: list[datetime]|None
    ) -> tuple[datetime, datetime, datetime, float, float, float, float]:
        #Get chat timestamps
        chat_first_timestamp = min(x for x in chat_ts) if chat_ts else None 
        chat_last_timestamp = max(x for x in chat_ts) if chat_ts else None     
        
        #Get last interaction
        chat_timestamps_list = chat_ts or []
        candidates = [x for x in [eval_ts] + chat_timestamps_list if x is not None]
        first_ts = min(candidates) if candidates else None
        last_ts = max(candidates) if candidates else None
        
        #Compute chat time_interval and full experiment time interval
        chat_first_last_time_interval = (
            (chat_last_timestamp - chat_first_timestamp).total_seconds() / 60
            if chat_first_timestamp and chat_last_timestamp
            else None
        )
        
        full_first_last_time_interval = (
            (last_ts - first_ts).total_seconds() / 60
            if chat_first_timestamp and chat_last_timestamp
            else None
        )
        
        #Compute the difference in minutes
        diffs_minutes = (
            [
                (chat_ts[i+1] - chat_ts[i]).total_seconds() / 60
                for i in range(len(chat_ts) - 1)
            ]
            if chat_ts and len(chat_ts) > 1
            else None
        )
        
        if diffs_minutes:
            mean_diff = statistics.mean(diffs_minutes)
            median_diff = statistics.median(diffs_minutes)
            stdev_diff = statistics.stdev(diffs_minutes) if len(diffs_minutes) > 1 else 0 
            mean_weighted_diff = mean_diff/(1+stdev_diff) 
            
            #Compute Gini Score for user interactions
            score = calculate_gini_chat_interaction_distribution_score(
                timestamps=chat_ts
            )
            
            #Multiply score by full_first_last_time_interval
            gini_chat_interaction_distribution = score * full_first_last_time_interval * len(chat_ts)       
        else: 
            mean_diff, median_diff, mean_weighted_diff, gini_chat_interaction_distribution = None, None, None, None
            
        return chat_first_timestamp, chat_last_timestamp, first_ts, last_ts, chat_first_last_time_interval, full_first_last_time_interval, mean_diff, median_diff, mean_weighted_diff, gini_chat_interaction_distribution

    rows = []

    for data_id, data in raw_data.items():

        eval_interactions = data['evaluation_interactions']
        eval_ts, chat_ts = _get_timestamps(data)

        if eval_interactions:
            answers = re.findall(
                r'Respuesta:\s*(.+?)(?=\n\nANSWER|\Z)',
                eval_interactions['user-answers'],
                re.DOTALL
            )
            eval_pass = eval_interactions['pass']
        else:
            answers  = []
            eval_pass = None
        
        chat_first_timestamp, chat_last_timestamp, first_ts, last_ts, chat_first_last_time_interval, full_first_last_time_interval, mean_diff, median_diff, mean_weighted_diff, gini_chat_interaction_distribution = process_interactions_timestamps(
            eval_ts=eval_ts,
            chat_ts=chat_ts
        )

        rows.append({
            'id':                       data_id,
            'chat_interactions_counts': len(data['chat_interactions']),
            'evaluation_answers_counts': len(answers),
            'evaluation_pass':          eval_pass,
            'chat_first_timestamp': chat_first_timestamp,
            'chat_last_timestamp': chat_last_timestamp,
            'tool_first_timestamp': first_ts,
            'tool_last_timestamp': last_ts,
            'chat_usage_time_interval': chat_first_last_time_interval,
            'chat_mean_time_per_interaction': mean_diff,
            'chat_median_time_per_interaction': median_diff,
            'chat_mean_weighted_time_per_interaction': mean_weighted_diff,
            'chat_attention_score': gini_chat_interaction_distribution,
            'tool_usage_time_interval': full_first_last_time_interval
        })

    interactions_data = pl.DataFrame(rows)

    interactions_data = interactions_data.with_columns(
        consultive_chat_used = pl.col('chat_interactions_counts') > 0,
        evaluator_chat_used = pl.col('evaluation_answers_counts') > 0
    ).with_columns(
        pl.when(pl.col("chat_interactions_counts") == 0)
        .then(pl.lit("No Usado"))
        .otherwise(
            pl.col("chat_interactions_counts")
            .qcut(3, labels=["Baja", "Media", "Alta"], allow_duplicates=True) # q33 and q67 are used
            .cast(pl.Utf8)
        )
        .alias("chat_freq_use")
    ).with_columns(
        pl.when(pl.col("chat_interactions_counts") == 0)
        .then(pl.lit("No Usado"))
        .otherwise(
            pl.col("chat_interactions_counts")
            .qcut(2, labels=["Baja", "Alta"], allow_duplicates=True) # q50 is used
            .cast(pl.Utf8)
        )
        .alias("chat_freq_use_v2")
    )
    
    #Normalize chat_attention_score 
    interactions_data = interactions_data.with_columns(
        chat_attention_score = (pl.col("chat_attention_score") - pl.col("chat_attention_score").min()) / 
                    (pl.col("chat_attention_score").max() - pl.col("chat_attention_score").min())
    )

    return interactions_data

#########################################################################################################################################################

def load_prompt():
    #Load prompt template
    with open(template_path, "r", encoding="utf-8") as f:
        original_prompt_template = f.read()
    
    #Load video transcription summary
    with open(summary_path, "r", encoding="utf-8") as f:
        video_summary = f.read()
    
    #Load full transcription
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = json.load(f)["vid_metadata_col"]["Subtitles"][0]["content"]
    
    #Apply string formatting to the prompt template
    prompt_template = original_prompt_template.format(
        summary=video_summary,
        #transcription=transcription,
        conversation_context="__CONVERSATION_CONTEXT__",
        user_interaction="__USER_INTERACTION__"
    )
    
    return prompt_template


#########################################################################################################################################################

def semantic_depth_index(client, model, temperature, prompt):

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=SemanticDepthOutput
      )
    )

    response = json.loads(response.text)

    return response

#########################################################################################################################################################

def generate_semantic_depth_index(client, model, temperature, raw_data, interaction_cheating_df, only_validation=False):

    semantic_depth_data = {}
    
    prompt_template = load_prompt()

    data_ids = list(raw_data.keys())
    total_ids = len(data_ids)
    
    cheating_interactions = interaction_cheating_df.filter(
        pl.col("cheating_score_llm") == 1
    )["user_input"].to_list()
    
    #Load validated sample
    with open(validation_sample_path, "r", encoding="utf-8") as f:
        validation_sample_data = json.load(f)
        validated_students = list(validation_sample_data.keys())

    for idx, data_id in enumerate(data_ids, start=1):
        if data_id not in validated_students and only_validation:
            continue

        user_interactions = raw_data[data_id]['chat_interactions']
        total_interactions = len(user_interactions)

        logging.info(f"Processing [{idx}/{total_ids}] id: {data_id} — {total_interactions} interactions")

        if user_interactions:
            semantic_depth_data[data_id] = []
            for i, interaction in enumerate(user_interactions, start=1):
                if interaction["user_input"] in cheating_interactions:
                    response = {
                        "reasoning_semantic_depth_level": "The user's question is identical to any static content question or it is directly soliciting answers to the quiz without any personal engagement with the material.",
                        "semantic_depth_level": 1
                    }
                    semantic_depth_data[data_id].append(interaction | response)
                    continue
                
                context = user_interactions[:i-1]
                
                prompt = prompt_template.replace("__CONVERSATION_CONTEXT__", str(context))
                prompt = prompt.replace("__USER_INTERACTION__", str(interaction))
                
                response = semantic_depth_index(
                    client=client, 
                    model=model, 
                    temperature=temperature, 
                    prompt=prompt
                )
                
                semantic_depth_data[data_id].append(interaction | response)
                logging.info(f"  -> [{i}/{total_interactions}] interactions processed")
        else:
            logging.warning(f"  -> No interactions found, skipping.")

    return semantic_depth_data

#########################################################################################################################################################

def categorize_wsdi(wsdi: float) -> str:
    if wsdi <= 1.4:
        return "Irrelevante" # <= 1.4
    elif wsdi <= 2.2:
        return "Superficial" # (1.4, 2.2]
    else:
        return "Profunda" # > 2.2

def categorize_wsdi_v2(wsdi: float) -> str:
    if wsdi <= 0.5:
        return 'Out_of_context' # <= 0.5
    elif wsdi <= 1.4:
        return "Cheating" # (0.5, 1.4]
    elif wsdi <= 2.2:
        return "Superficial" # (1.4, 2.2]
    else:
        return "Profunda" # > 2.2
    
#########################################################################################################################################################

def categorize_high_quality(wsdi: float) -> str:
    if wsdi >= 2:
        return True  
    else:
        return False
    
#########################################################################################################################################################

def process_semantic_depth_data(semantic_depth_data):

    rows = [
        {'id': id_, 'semantic_depth_level': q['semantic_depth_level']}
        for id_, questions in semantic_depth_data.items()
        for q in questions
    ]

    semantic_depth_df = pl.DataFrame(rows)

    wsdi_df = (
        semantic_depth_df
        # Paso 1: contar nij (preguntas por alumno y nivel)
        .group_by(['id', 'semantic_depth_level'])
        .agg(pl.len().alias('n_ij'))
        # Paso 2: calcular wj * nij
        .with_columns(
            (pl.col('semantic_depth_level') * pl.col('n_ij')).alias('w_x_n')
        )
        # Paso 3: agregar por alumno → Σ(wj * nij) y N_total
        .group_by('id')
        .agg([
            pl.col('w_x_n').sum().alias('weighted_sum'),
            pl.col('n_ij').sum().alias('N_total')
        ])
        # Paso 4: calcular WSDI
        .with_columns(
            (pl.col('weighted_sum') / pl.col('N_total')).alias('WSDI')
        )
        .sort('id')
    )

    # Paso 5: pivot → count_level_X por alumno, luego join a wsdi_df
    rename_map = {
        '0': 'count_out_of_context',
        '1': 'count_cheating',
        '2': 'count_superficial',
        '3': 'count_deep'
     }

    count_df = (
        semantic_depth_df
        .group_by(['id', 'semantic_depth_level'])
        .agg(pl.len().alias('n_ij'))
        .pivot(on='semantic_depth_level', index='id', values='n_ij', aggregate_function='first')
        .rename(rename_map)
        .fill_null(0)
        .sort('id')
    )

    wsdi_df = wsdi_df.join(count_df, on='id', how='left')

    return semantic_depth_df, wsdi_df

#########################################################################################################################################################

def add_wsdi_cheating_score(wsdi_df, cheating_df, interactions_df):

    interactions_df = interactions_df.join(
        wsdi_df[['id', 'WSDI', 'count_cheating', 'count_out_of_context', 'count_superficial', 'count_deep']],
        how='left',
        on='id'
    ).with_columns(
        pl.col("WSDI").map_elements(
            categorize_wsdi, 
            return_dtype=pl.String
        ).alias("WSDI_cat")
    ).with_columns(
        pl.col("WSDI").map_elements(
            categorize_wsdi_v2, 
            return_dtype=pl.String
        ).alias("WSDI_cat_v2")
    ).with_columns(
        pl.col("WSDI").map_elements(
            categorize_high_quality, 
            return_dtype=pl.Boolean
        ).alias("high_quality_use")
    ).with_columns(
        pl.when(pl.col('WSDI_cat').is_null()).
        then(pl.col('chat_freq_use')).
        otherwise(pl.col('WSDI_cat')).
        alias('WSDI_cat')
    ).with_columns(
        pl.when(pl.col('WSDI_cat_v2').is_null()).
        then(pl.col('chat_freq_use')).
        otherwise(pl.col('WSDI_cat_v2')).
        alias('WSDI_cat_v2')
    ).with_columns(
        pl.when(pl.col('high_quality_use').is_null()).
        then(pl.col('chat_freq_use')).
        otherwise(pl.col('high_quality_use')).
        alias('high_quality_use')
    )

    interactions_df = interactions_df.join(
        cheating_df.select(['id', 'cheating_score_llm']), 
        how='left', 
        on='id'
    )

    return interactions_df

#########################################################################################################################################################

def segment_experimental_type(interactions_df):
    """
    Segmenta el grupo experimental en base a Frecuencia y Calidad de uso.
    
    - AA: Frecuencia Alta / Calidad Alta
    - BA: Frecuencia no-Alta / Calidad Alta
    - AB: Frecuencia Alta / Calidad Baja
    - BB: Frecuencia no-Alta / Calidad Baja
    """
    freq_alta = pl.col("chat_freq_use") == "Alta"
    freq_baja = pl.col("chat_freq_use") == "Baja"
    freq_media = pl.col("chat_freq_use") == "Media"

    freq_alta_v2 = pl.col("chat_freq_use_v2") == "Alta"
    freq_baja_v2 = pl.col("chat_freq_use_v2") == "Baja"

    freq_not_used = pl.col("chat_freq_use") == "No Usado"

    calidad_alta = pl.col("high_quality_use") == True
    calidad_baja = pl.col("high_quality_use") == False

    interactions_df =  interactions_df.with_columns(
        pl.when( freq_alta &  calidad_alta).then(pl.lit("ExpAA"))
          .when(~freq_alta &  calidad_alta).then(pl.lit("ExpBA"))
          .when( freq_alta & ~calidad_alta).then(pl.lit("ExpAB"))
          .otherwise(pl.lit("ExpBB"))
          .alias("experimental_type_v1")
    )

    interactions_df =  interactions_df.with_columns(
        pl.when( freq_alta &  calidad_alta).then(pl.lit("ExpAA"))
          .when( freq_baja &  calidad_alta).then(pl.lit("ExpBA"))
          .when( freq_alta &  calidad_baja).then(pl.lit("ExpAB"))
          .when( freq_baja &  calidad_baja).then(pl.lit("ExpBB"))
          .when( freq_media & calidad_alta).then(pl.lit("ExpMA"))
          .when( freq_media & calidad_baja).then(pl.lit("ExpMB"))
          .when(freq_not_used).then(pl.lit("ExpNotUsed"))
          .otherwise(pl.lit("ExpOther"))
          .alias("experimental_type_v2")
    )

    interactions_df =  interactions_df.with_columns(
        pl.when( freq_alta_v2 &  calidad_alta).then(pl.lit("ExpAA"))
          .when( freq_baja_v2 &  calidad_alta).then(pl.lit("ExpBA"))
          .when( freq_alta_v2 &  calidad_baja).then(pl.lit("ExpAB"))
          .when( freq_baja_v2 &  calidad_baja).then(pl.lit("ExpBB"))
          .when(freq_not_used).then(pl.lit("ExpNotUsed"))
          .otherwise(pl.lit("ExpOther"))
          .alias("experimental_type_v3")
    )

    interactions_df =  interactions_df.with_columns(
        pl.when(calidad_alta).then(pl.lit("ExpA"))
          .when(calidad_baja).then(pl.lit("ExpB"))
          .when(freq_not_used).then(pl.lit("ExpNotUsed"))
          .otherwise(pl.lit("ExpOther"))
          .alias("experimental_type_v4")
    )

    interactions_df =  interactions_df.with_columns(
        pl.when(freq_not_used).then(pl.lit("ExpNotUsed"))
          .otherwise(pl.lit("ExpUsed"))
          .alias("experimental_type_v5")
    )

    interactions_df =  interactions_df.with_columns(
        pl.when( freq_alta_v2 ).then(pl.lit("ExpA"))
        .otherwise(pl.lit("ExpB"))
          .alias("experimental_type_v6")
    )

    return interactions_df

#########################################################################################################################################################

def analizar_rendimiento_por_interaccion(df_parquet, df_interacciones):
    """
    Cruza los resultados de aprendizaje con las categorías de interacción del chatbot.
    
    Args:
        df_parquet: DataFrame de Polars con los resultados cruzados (Pre/Post/Hake).
        df_interacciones: DataFrame con las columnas ['id', 'WSDI_cat'].
    """
    
    # 1. Aseguramos el cruce por ID para tener la categoría WSDI junto a las notas
    # Seleccionamos solo las columnas necesarias de interacciones para no ensuciar
    df_merged = df_parquet.join(
        df_interacciones.select(["id", "WSDI_cat"]), 
        on="id", 
        how="left"
    )

    # 2. Definimos las métricas que queremos resumir
    # Usamos las ganancias de Hake que ya tienes en tu Parquet
    metricas = [
        "puntuacion_tc_hake_gain",
        "puntuacion_tc_retencion_hake_gain",
        "puntuacion_tc_transferencia_hake_gain",
        "puntuacion_tc_post",
        "puntuacion_tcc_rel_post" # Carga cognitiva relevante (si quieres verla)
    ]

    # 3. Realizamos la agregación por categoría de WSDI
    resumen = (
        df_merged
        .filter(pl.col("WSDI_cat").is_not_null()) # Quitamos alumnos sin interacción (Control)
        .group_by("WSDI_cat")
        .agg([
            pl.count("id").alias("n_alumnos"),
            *[pl.col(m).mean().round(3).alias(f"mean_{m}") for m in metricas],
            *[pl.col(m).std().round(3).alias(f"std_{m}") for m in metricas]
        ])
        .sort("mean_puntuacion_tc_hake_gain", descending=True)
    )

    return resumen, df_merged
