import polars as pl
import os

script_path = os.getcwd()
project_path = os.path.join(script_path, '..', '..', '..')
data_filename = 'interactions_processed_data.parquet'
data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', data_filename)
data_dir = os.path.join(project_path, 'data', 'forms')
centros_data_path = os.path.join(project_path, 'data', 'centros', 'renta_centros_educativos.csv')
processed_data_dir = os.path.join(data_dir, 'processed_data')
processed_forms_path = os.path.join(processed_data_dir, 'processed_pre_post_forms.parquet')

processed_forms_data = pl.read_parquet(processed_forms_path)

renta = pl.read_csv(centros_data_path)

# Mapear los nombres del CSV a las claves
mapeo = {
    "IES Ramiro de Maeztu (Madrid)":                  "RamiroMaeztu",
    "IES Laguna de Joatzel (Getafe)":                 "Laguna",
    "Colegio Jesús María - García Noblejas (Madrid)": "JesusMaria",
    "IES José García Nieto (Las Rozas)":              "GarciaNieto",
}
renta = renta.with_columns(
    pl.col("centro").replace(mapeo)
)

# Join
processed_forms_data = processed_forms_data.join(renta,
                                                  on="centro",
                                                   how="left")

print(processed_forms_data.head(5))