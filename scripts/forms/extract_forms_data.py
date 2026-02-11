import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import os

# 1. Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_to_json = os.path.join(BASE_DIR, 'credentials.json')

# 2. Definición de permisos (Scope)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# 3. Autenticación
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name(path_to_json, scope)
    client = gspread.authorize(creds)
    print("✓ Autenticación exitosa.")
except Exception as e:
    print(f"× Error en credenciales: {e}")
    exit()

# 4. Función de descarga individual
def download_data(nombre_hoja, nombre_archivo_csv):
    try:
        # Abrir la hoja de cálculo
        spreadsheet = client.open(nombre_hoja)
        sheet = spreadsheet.get_worksheet(0)
        
        # Convertir a DataFrame
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Guardar localmente
        ruta_guardado = os.path.join(BASE_DIR, nombre_archivo_csv)
        df.to_csv(ruta_guardado, index=False)
        
        print(f"✓ Hoja '{nombre_hoja}' guardada correctamente en: {nombre_archivo_csv}")
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"× Error: No se encontró la hoja '{nombre_hoja}'")
    except Exception as e:
        print(f"× Error inesperado con '{nombre_hoja}': {e}")

# 5. Ejecución de las descargas (Por separado)
df_pre = download_data("Cuestionario Previo (respuestas)", "respuestas_pre.csv")
df_post = download_data("Cuestionario Post (respuestas)", "respuestas_post.csv")
