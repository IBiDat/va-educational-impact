###########################################################################################

import os, json, logging
import polars as pl

###########################################################################################

def process_school_hashes(json_path: str) -> pl.DataFrame:
    """
    Lee un JSON de hashes, extrae el nombre del centro y asigna
    aleatoriamente 50% Control y 50% Experimental.
    """
    try:
        # 1. Cargar JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        id_list = data.get('hashes', [])
        
        if not id_list:
            logging.warning(f"⚠️  El archivo {os.path.basename(json_path)} no contiene hashes.")
            return None

        # 2. Crear DataFrame con Polars
        df = pl.DataFrame({"id": id_list})

        # 3. Extraer el nombre del centro desde el propio ID (ej: 'JoseGarciaNieto-9ba' -> 'JoseGarciaNieto')
        # Asumimos que el separador es siempre un guion '-'
        df = df.with_columns(
            pl.col("id").str.split("-").list.get(0).alias("centro")
        )

        # 4. Asignar Grupos (50/50)
        # Barajamos aleatoriamente para que la asignación no dependa del orden de llegada
        df = df.sample(fraction=1.0, shuffle=True, seed=42)
        
        # Usamos el índice de fila para dividir
        df = df.with_row_index("row_idx")
        total_rows = df.height
        cutoff = total_rows // 2  # División entera

        df = df.with_columns(
            pl.when(pl.col("row_idx") < cutoff)
            .then(pl.lit("control"))
            .otherwise(pl.lit("experimental"))
            .alias("grupo")
        )

        # 5. Seleccionar y ordenar columnas finales
        df_final = df.select(["centro", "id", "grupo"])
        
        return df_final

    except Exception as e:
        logging.error(f"❌ Error processing file {json_path}: {e}")
        return None
    
###########################################################################################