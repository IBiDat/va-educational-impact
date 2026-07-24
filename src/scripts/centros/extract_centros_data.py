'''
renta_centros.py
================
Genera renta_centros_educativos.csv con la renta media por hogar (Atlas de
Distribución de Renta de los Hogares del INE, año 2023) de la sección censal
en la que se ubica cada uno de estos centros:

  - IES José García Nieto      (Las Rozas)
  - IES Ramiro de Maeztu       (Madrid)
  - Colegio Jesús María        (C/ Hnos. García Noblejas 68, Madrid)
  - IES Laguna de Joatzel      (Getafe)

Flujo:
  1. Geocodifica las direcciones con Nominatim (OpenStreetMap, gratuito).
  2. Descarga el shapefile de secciones censales del INE (≈63 MB).
  3. Spatial join punto-en-polígono → CUSEC (código de sección censal).
  4. Descarga el CSV del Atlas de Renta del INE (≈340 MB) y lo filtra.
  5. Exporta renta_centros_educativos.csv y .xlsx.

Requisitos:
  pip install pandas geopandas geopy requests openpyxl
'''

import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
output_dir = os.path.join(project_path, 'data', 'centros')
os.makedirs(output_dir, exist_ok=True)

URL_SHP   = "https://www.ine.es/prodyser/cartografia/seccionado_2024.zip"
URL_RENTA = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/30824.csv?nocab=1"

ZIP_SHP   = os.path.join(output_dir, "seccionado_2024.zip")
DIR_SHP   = os.path.join(output_dir, "seccionado")
SHP_FILE  = os.path.join(DIR_SHP, "SECC_CE_20240101.shp")
CSV_RENTA = os.path.join(output_dir, "atlas_renta_30824.csv")

OUT_CSV   = os.path.join(output_dir, "centros_data.csv")

# Centros actualizados con el curso
CENTROS = [
    ("IES José García Nieto (Las Rozas)",
        "Calle Camilo José Cela 24, 28232 Las Rozas de Madrid, España", "4ºESO"),
    ("IES Ramiro de Maeztu (Madrid)",
        "Calle Serrano 127, 28006 Madrid, España", "1ºBACHILLERATO"),
    ("Colegio Jesús María - García Noblejas (Madrid)",
        "Calle Hermanos García Noblejas 68, 28037 Madrid, España", "1ºBACHILLERATO"),
    ("IES Laguna de Joatzel (Getafe)",
        "Avenida de las Vascongadas s/n, 28903 Getafe, España", "1ºFP-MEDIO"),
]

AÑO = 2023

# ─── 1. GEOCODIFICACIÓN ──────────────────────────────────────────────────────
def geocodifica(centros):
    print("→ Geocodificando con Nominatim (≈1s por school)…")
    geo = Nominatim(user_agent="renta_centros_educativos")
    geocode = RateLimiter(geo.geocode, min_delay_seconds=1.2)

    filas = []
    for nombre, direccion, curso in centros:
        loc = geocode(direccion, country_codes="es")
        if loc is None:
            print(f"  ✗ {nombre}: no localizado")
            filas.append({"nombre": nombre, "direccion": direccion, "curso": curso,
                          "lat": None, "lon": None})
        else:
            print(f"  ✓ {nombre} → ({loc.latitude:.5f}, {loc.longitude:.5f})")
            filas.append({"nombre": nombre, "direccion": direccion, "curso": curso,
                          "lat": loc.latitude, "lon": loc.longitude})
    return pd.DataFrame(filas)

# ─── 2. DESCARGA SHAPEFILE ───────────────────────────────────────────────────
def descarga_shapefile():
    if os.path.exists(SHP_FILE):
        return
    print("→ Descargando shapefile del INE…")
    r = requests.get(URL_SHP, timeout=120)
    with open(ZIP_SHP, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(ZIP_SHP) as z:
        z.extractall(DIR_SHP)

# ─── 3. ASIGNACIÓN DE SECCIÓN CENSAL ────────────────────────────────────────
def asigna_seccion(df_centros):
    print("→ Cargando seccionado…")
    secc = gpd.read_file(SHP_FILE).to_crs("EPSG:4326")
    secc = secc[["CUSEC", "NMUN", "NPRO", "geometry"]]
    centros_gdf = gpd.GeoDataFrame(
        df_centros.dropna(subset=["lat", "lon"]),
        geometry=gpd.points_from_xy(df_centros["lon"], df_centros["lat"]),
        crs="EPSG:4326",
    )
    out = gpd.sjoin(centros_gdf, secc, how="left", predicate="within")
    return out.drop(columns=["geometry", "index_right"])

# ─── 4. DESCARGA ATLAS DE RENTA ──────────────────────────────────────────────
def descarga_renta():
    if os.path.exists(CSV_RENTA):
        return
    print("→ Descargando Atlas de Renta del INE (340 MB)…")
    with requests.get(URL_RENTA, timeout=600, stream=True) as r:
        with open(CSV_RENTA, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

def renta_para_cusecs(cusecs_set, año=AÑO):
    indicadores_deseados = [
        "Renta neta media por hogar",
        "Renta neta media por persona",
        "Mediana de la renta por unidad de consumo",
    ]
    trozos = []
    for chunk in pd.read_csv(CSV_RENTA, sep=";", encoding="utf-8",
                             chunksize=200_000, low_memory=False):
        chunk["cusec"] = chunk["Secciones"].astype(str).str.extract(r"^(\d{10})")
        m = (
            chunk["cusec"].isin(cusecs_set)
            & (chunk["Periodo"] == año)
            & (chunk["Indicadores de renta media"].isin(indicadores_deseados))
        )
        if m.any():
            trozos.append(chunk[m])

    df = pd.concat(trozos, ignore_index=True)
    pivot = (
        df.pivot_table(
            index="cusec",
            columns="Indicadores de renta media",
            values="Total",
            aggfunc="first",
        )
        .reset_index()
    )
    for col in pivot.columns:
        if col != "cusec":
            pivot[col] = (
                pivot[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
    return pivot

# ─── 5. ORQUESTACIÓN ─────────────────────────────────────────────────────────
def main():
    df_geo = geocodifica(CENTROS)
    descarga_shapefile()
    df_seccion = asigna_seccion(df_geo)
    descarga_renta()
    df_renta = renta_para_cusecs(set(df_seccion["CUSEC"].dropna()))

    final = df_seccion.merge(df_renta, left_on="CUSEC", right_on="cusec", how="left")
    final = final.drop(columns=["cusec"]).rename(columns={
        "CUSEC": "seccion_censal",
        "NMUN":  "municipio",
        "NPRO":  "provincia",
        "Renta neta media por hogar":                   "renta_media_hogar_eur",
        "Renta neta media por persona":                 "renta_media_persona_eur",
        "Mediana de la renta por unidad de consumo":    "mediana_renta_uc_eur",
    })

    final = final.sort_values("renta_media_hogar_eur", ascending=False)
    final.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 100)
    print(f"Resultado (Atlas de Renta INE, año {AÑO}):")
    print("=" * 100)
    print(final.to_string(index=False))

if __name__ == "__main__":
    main()