import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..','..')
sys.path.append(project_path)
from src.utils.forms.forms_data_extraction import get_gspread_client, get_forms_data

# Configuración de rutas locales
DATA_DIR = os.path.join(project_path, 'data')
PATH_JSON = os.path.join(DATA_DIR, 'credentials.json')
FORMS_DIR = os.path.join(project_path, 'data', 'forms')

# Crear carpeta de datos si no existe
if not os.path.exists(FORMS_DIR):
    os.makedirs(FORMS_DIR)

def run_extraction():
    # 1. Obtener cliente
    client = get_gspread_client(PATH_JSON)
    
    if client:
        # 2. Definir descargas
        formularios = {
            "Cuestionario Previo (respuestas)": "respuestas_pre.csv",
            "Cuestionario Post (respuestas)": "respuestas_post.csv"
        }
        
        # 3. Ejecutar descargas en bucle
        for nombre_hoja, nombre_csv in formularios.items():
            ruta_csv = os.path.join(FORMS_DIR, nombre_csv)
            get_forms_data(client, nombre_hoja, ruta_csv)

if __name__ == "__main__":
    run_extraction()