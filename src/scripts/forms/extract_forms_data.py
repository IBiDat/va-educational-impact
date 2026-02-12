###########################################################################################

# --- IMPORTS ---

import os, sys, logging

###########################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths (using __file__ for reliability instead of getcwd)
script_path = os.path.dirname(os.path.abspath(__file__))
# Ajuste de rutas relativas según tu script original 
project_path = os.path.join(script_path, '..', '..', '..')
sys.path.append(project_path)

# Data directories
forms_data_dir = os.path.join(project_path, 'data', 'forms', 'raw_data')

# Configuration Files
credentials_json_path = os.path.join(forms_data_dir, 'credentials.json')

# Output Directories
# Ensure output directory exists
os.makedirs(forms_data_dir, exist_ok=True)

###########################################################################################

# --- IMPORTS (PROJECT MODULES) ---

# Importamos módulos propios después de ajustar el sys.path

from src.utils.forms.forms_data_extraction import (
    get_gspread_client, 
    get_forms_data
)

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow for Google Forms Data Extraction.
    Downloads specific form responses from Google Sheets to CSV.
    """
    logging.info("▶️ STARTING GOOGLE FORMS DATA EXTRACTION PIPELINE")

    # 1. Authenticate with Google API
    logging.info("STEP 1: Authenticating with Google Sheets API...\n")
    
    client = None
    try:
        if not os.path.exists(credentials_json_path):
            raise FileNotFoundError(f"Credentials file not found at: {credentials_json_path}")

        client = get_gspread_client(credentials_json_path)
        
        if not client:
            raise ConnectionError("Failed to initialize gspread client (client is None).")
            
        logging.info(" -> Authentication successful.\n")

    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        sys.exit(1)

    # 2. Define Download Tasks
    logging.info("STEP 2: Defining download tasks...\n")
    
    formularios = {
        "Cuestionario Previo (respuestas)": "respuestas_forms_pre.csv",
        "Cuestionario Post (respuestas)": "respuestas_forms_post.csv"
    }
    
    logging.info(f" -> Found {len(formularios)} forms to process.\n")

    # 3. Execute Downloads
    logging.info("STEP 3: Processing form downloads...\n")

    success_count = 0
    
    for nombre_hoja, nombre_csv in formularios.items():
        try:
            output_file_path = os.path.join(forms_data_dir, nombre_csv)
            logging.info(f" -> Downloading: '{nombre_hoja}' ...")
            
            # Llamada a la función de extracción
            get_forms_data(client, nombre_hoja, output_file_path)
            
            # Verificación simple de que el archivo se creó
            if os.path.exists(output_file_path):
                logging.info(f"    ✅ Saved to: {os.path.basename(output_file_path)}")
                success_count += 1
            else:
                logging.warning(f"    ⚠️ Function completed but file was not found: {output_file_path}")

        except Exception as e:
            logging.error(f"    ❌ Error processing '{nombre_hoja}': {e}")
            # No hacemos exit(1) aquí para intentar descargar los siguientes formularios

    # 4. Final Summary
    logging.info("\n-----------------------------------------------------------")
    if success_count == len(formularios):
        logging.info("✅ PIPELINE FINISHED SUCCESSFULLY: All forms downloaded.")
    else:
        logging.warning(f"⚠️ PIPELINE FINISHED WITH WARNINGS: {success_count} out of {len(formularios)} downloads successful.")

###########################################################################################

if __name__ == "__main__":
    main()