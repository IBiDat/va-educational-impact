"""
Anonymise processed_forms_interactions_data.parquet for public release.

- `id`      -> sequential record index (the original id embeds the school name)
- `school`  -> school_1 ... school_4, via the fixed mapping below
- geographic / address columns identifying each centre are dropped
"""

import pandas as pd

IN_PATH = "data/combined/processed_forms_interactions_data.parquet"
OUT_PATH = "experiment_resources/public_forms_interactions_data.parquet"

# ---------------------------------------------------------------- mapping ---
# Fixed, explicit mapping. Edit here if the agreed order differs.
SCHOOL_MAP = {
    "RamiroMaeztu":    "school_1",   
    "JoseGarciaNieto": "school_2",  
    "JesusMaria":      "school_3",   
    "Laguna":          "school_4",  
}

# Columns that identify the centre (address, coordinates, census tract and the
# census-tract income variables derived from it).
IDENTIFYING_COLS = [
    "nombre", "direccion", "lat", "lon",
    "seccion_censal", "municipio", "provincia",
]
# Set to True to also drop the census-tract income variables: they are a
# near-unique fingerprint of the school's location.
DROP_INCOME = True
INCOME_COLS = [
    "mediana_renta_uc_eur", "renta_media_hogar_eur", "renta_media_persona_eur",
]

# Save the id -> record correspondence outside the repository if you need it
# internally. Set to None to not write it at all.
KEY_PATH = None  # e.g. "../private/id_record_key.csv"

# ------------------------------------------------------------------ script --
df = pd.read_parquet(IN_PATH)

# 1. school -------------------------------------------------------------------
unknown = set(df["school"].dropna().unique()) - set(SCHOOL_MAP)
if unknown:
    raise ValueError(f"School values not present in SCHOOL_MAP: {sorted(unknown)}")
df["school"] = df["school"].map(SCHOOL_MAP)

# 2. id -----------------------------------------------------------------------
# Sort by the anonymised school first so that the record order carries no
# information about the original (alphabetical) id ordering, then renumber.
df = (df.sort_values(["school", "id"], kind="mergesort")
        .reset_index(drop=True))

if KEY_PATH is not None:
    (df[["id"]].reset_index().rename(columns={"index": "record"})
       .to_csv(KEY_PATH, index=False))

df = df.drop(columns=["id"])
df.insert(0, "record", range(len(df)))

# 3. centre-identifying columns ----------------------------------------------
to_drop = [c for c in IDENTIFYING_COLS if c in df.columns]
if DROP_INCOME:
    to_drop += [c for c in INCOME_COLS if c in df.columns]
df = df.drop(columns=to_drop)

# 4. checks -------------------------------------------------------------------
leak = [c for c in df.columns
        if df[c].dtype == object
        and df[c].astype(str).str.contains("|".join(SCHOOL_MAP), na=False).any()]
assert not leak, f"Original school names still present in: {leak}"

df.to_parquet(OUT_PATH, index=False)

print(f"{df.shape[0]} rows, {df.shape[1]} columns written to {OUT_PATH}")
print(df["school"].value_counts().sort_index().to_string())
print("dropped:", to_drop)