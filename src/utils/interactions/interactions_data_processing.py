#########################################################################################################################################################

import os
import logging
import json
import re
import polars as pl
from google.genai import types
from typing import TypedDict

class SemanticDepthOutput(TypedDict):
    reasoning_semantic_depth_level: str
    semantic_depth_level: int

#########################################################################################################################################################

def process_interactions_data(raw_data):

    evaluation_answers, evaluation_pass = [], []

    data_ids = list(raw_data.keys())

    for data_id in data_ids:
        if raw_data[data_id]['evaluation_interactions']:
            evaluation_interactions = raw_data[data_id]['evaluation_interactions']['user-answers']
            evaluation_pass.append(raw_data[data_id]['evaluation_interactions']['pass'])
            evaluation_answers.append(re.findall(r'Respuesta:\s*(.+?)(?=\n\nANSWER|\Z)', evaluation_interactions, re.DOTALL))
        else:
            evaluation_answers.append([])
            evaluation_pass.append(None)
            
    evaluation_answers_counts = [len(x) for x in evaluation_answers]

    chat_interactions_counts = [len(raw_data[data_id]['chat_interactions']) for data_id in data_ids]

    interactions_data = pl.DataFrame({
        'id': data_ids, 
        'chat_interactions_counts': chat_interactions_counts,
        'evaluation_answers_counts': evaluation_answers_counts,
        'evaluation_pass': evaluation_pass
    })

    interactions_data = interactions_data.with_columns(
        consultive_chat_used = pl.col('chat_interactions_counts') > 0,
        evaluator_chat_used = pl.col('evaluation_answers_counts') > 0
    ).with_columns(
        pl.when(pl.col("chat_interactions_counts") == 0)
        .then(pl.lit("No Usado"))
        .otherwise(
            pl.col("chat_interactions_counts")
            .qcut(3, labels=["Baja", "Media", "Alta"], allow_duplicates=True)
            .cast(pl.Utf8)
        )
        .alias("chat_freq_use")
    )

    return interactions_data

#########################################################################################################################################################

def semantic_depth_index(client, model, temperature, user_interaction):

    prompt = f"""
    You are an expert educational interaction analyst specializing in Self-Regulated Learning (SRL) and learning analytics.

    Your task is to assign a **Semantic Depth Level** from **0 (Null)** to **3 (Deep)** to the provided student question, based on the cognitive operation required to formulate it.

    ---
    **SEMANTIC DEPTH SCALE (0-3):**
    * **3 - Deep:** Transfer, inference, or metacognition. The student applies concepts to new contexts, draws conclusions, or monitors their own understanding ("How does X apply to Y?", "What would happen if X?").
    * **2 - Intermediate:** Elaboration, paraphrasing, or simple relationships. The student seeks to understand causes, mechanisms, or alternative explanations ("Why does X happen?", "Can you explain X in a different way?").
    * **1 - Superficial:** Factual questions, definitions, or localization. The student retrieves isolated information ("What is X?", "When did X happen?", "Where is X defined?").
    * **0 - Not Relevant:** Questions not relevant to learning. Off-topic, social, or technical issues unrelated to the academic content.

    ---
    **THEORETICAL FRAMEWORK:**
    This scale is grounded in Zimmerman (2000) and Pintrich (2000) Self-Regulated Learning models, and operationalized through the Anderson & Krathwohl (2001) revised cognitive taxonomy.

    ---
    **OUTPUT FORMAT:**
    Return a single JSON object.
    Keys:
    - "reasoning_semantic_depth_level": A concise explanation (1-2 sentences). Step 1: Identify the cognitive operation required (recall, explanation, application, reflection). Step 2: Apply the scale to justify the level.
    - "semantic_depth_level": The integer level (0-3).

    Example:
    {{
      "reasoning_semantic_depth_level": "The student asks how a statistical concept applies to a real-world scenario, requiring transfer and inference beyond mere recall.",
      "semantic_depth_level": 3
    }}

    **USER INTERACTION TO CLASSIFY:**
    {user_interaction}
    """

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

def generate_semantic_depth_index(client, model, temperature, raw_data):

    semantic_depth_data = {}

    data_ids = list(raw_data.keys())

    for data_id in data_ids:
        user_interactions = raw_data[data_id]['chat_interactions']
        if user_interactions:
            semantic_depth_data[data_id] = []
            for interaction in user_interactions:
                response = semantic_depth_index(
                    client=client, 
                    model=model, 
                    temperature=temperature, 
                    user_interaction=interaction
                )
                semantic_depth_data[data_id].append(response)

    return semantic_depth_data

def generate_semantic_depth_index(client, model, temperature, raw_data):

    semantic_depth_data = {}

    data_ids = list(raw_data.keys())
    total_ids = len(data_ids)

    for idx, data_id in enumerate(data_ids, start=1):
        user_interactions = raw_data[data_id]['chat_interactions']
        total_interactions = len(user_interactions)

        logging.info(f"Processing [{idx}/{total_ids}] id: {data_id} — {total_interactions} interactions")

        if user_interactions:
            semantic_depth_data[data_id] = []
            for i, interaction in enumerate(user_interactions, start=1):
                response = semantic_depth_index(
                    client=client, 
                    model=model, 
                    temperature=temperature, 
                    user_interaction=interaction
                )
                semantic_depth_data[data_id].append(response)
                logging.info(f"  -> [{i}/{total_interactions}] interactions processed")
        else:
            logging.warning(f"  -> No interactions found, skipping.")

    return semantic_depth_data

#########################################################################################################################################################

def categorize_wsdi(wsdi: float) -> str:
    if wsdi <= 0.5:
        return "No Relevante" # < 0.5
    elif wsdi < 1.5:
        return "Superficial" # [0.5, 1.5)
    elif wsdi < 2.5:
        return "Intermedia" # [1.5, 2.5)
    else:
        return "Profunda" # >= 2.5

#########################################################################################################################################################

def process_semantic_depth_data(semantic_depth_data):

    rows = [
        {'id': id_, 'semantic_depth_level': q['semantic_depth_level']}
        for id_, questions in semantic_depth_data.items()
        for q in questions
    ]

    semantic_depth_df = pl.DataFrame(rows)

    weighted_semantic_depth_df = (
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

    weighted_semantic_depth_df = weighted_semantic_depth_df.with_columns(
        pl.col("WSDI").map_elements(categorize_wsdi, return_dtype=pl.String).alias("WSDI_cat")
    )

    return semantic_depth_df, weighted_semantic_depth_df

#########################################################################################################################################################

def process_combined_interactions_data(interactions_df):

    interactions_df = interactions_df.with_columns(
            high_quality_use = pl.col('WSDI_cat').is_in([
                'Intermedia', 
                'Profunda'
            ]),
        ).with_columns(
            pl.col('high_quality_use').replace(None, False)
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
    calidad_alta = pl.col("high_quality_use")

    interactions_df = interactions_df.with_columns(
        pl.when( freq_alta &  calidad_alta).then(pl.lit("AA"))
          .when(~freq_alta &  calidad_alta).then(pl.lit("BA"))
          .when( freq_alta & ~calidad_alta).then(pl.lit("AB"))
          .otherwise(pl.lit("BB"))
          .alias("experimental_type_freq_quality")
    )

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
