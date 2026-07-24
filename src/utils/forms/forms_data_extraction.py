###########################################################################################

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

###########################################################################################

def get_gspread_client(path_to_json):
    """Establece la conexión con Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(path_to_json, scope)
        client = gspread.authorize(creds)
        print("✓ Autenticación con Google exitosa.")
        return client
    except Exception as e:
        print(f"× Error en la autenticación: {e}")
        return None

def get_forms_data(client, nombre_hoja, ruta_salida):
    """Descarga una hoja específica y la guarda en CSV."""
    try:
        spreadsheet = client.open(nombre_hoja)
        sheet = spreadsheet.get_worksheet(0)
        
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        df.rename(columns={"Marca temporal": "timestamp"}, inplace=True)
        
        df.to_csv(ruta_salida, index=False)
        print(f"✓ Hoja '{nombre_hoja}' descargada en: {ruta_salida}")
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"× Error: No se encontró la hoja '{nombre_hoja}'")
    except Exception as e:
        print(f"× Error inesperado: {e}")
    return None

###########################################################################################