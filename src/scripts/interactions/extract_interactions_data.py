import json
from pymongo import MongoClient
import os

# --- 1. CONFIGURACIÓN GENERAL ---
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "universia_VA"

# Configuración del archivo de salida
CARPETA_DESTINO = "../data"
NOMBRE_ARCHIVO = "interactions_data.json"

# --- 2. CONEXIÓN A MONGODB ---
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("Conexión exitosa a la base de datos.")
except Exception as e:
    print(f"Error al conectar a MongoDB: {e}")
    exit()

interactions_col = db['interactions']
hashes_col = db['access_hashes']

# --- 3. LÓGICA DE EXTRACCIÓN CON FILTRO ---

interactions_data = {}

# Paso A: Obtener usuarios con 'used': True
query_users = {"used": True}
users_cursor = hashes_col.find(query_users)
count_users = hashes_col.count_documents(query_users)

print(f"Procesando {count_users} usuarios con 'used': True ...")

for user_doc in users_cursor:
    user_id = user_doc.get('user_id')
    participant_hash = user_doc.get('hash') 
    
    if not user_id or not participant_hash:
        continue 

    # --- CAMBIO AQUÍ ---
    # Inicializamos la estructura SIN repetir el 'id_participante' dentro.
    # La clave del diccionario ya nos dice quién es.
    interactions_data[participant_hash] = {
        'chat_interactions': [],
        'evaluation_interactions': {} 
    }

    # Paso B: Buscar interacciones
    user_interactions = interactions_col.find({'user_id': user_id}).sort('timestamp', 1)

    for interaction in user_interactions:
        i_type = interaction.get('type')

        if i_type == 'chat_interaction':
            chat_entry = {
                'user_input': interaction.get('user_input', ''),
                'model_response': interaction.get('model_response', '')
            }
            interactions_data[participant_hash]['chat_interactions'].append(chat_entry)

        elif i_type == 'evaluation':
            eval_entry = {
                'evaluation_questions': interaction.get('evaluation_questions', ''),
                'user-answers': interaction.get('user_answers', '')
            }
            interactions_data[participant_hash]['evaluation_interactions'] = eval_entry

# --- 4. GUARDADO SEGURO ---

# Crear carpeta si no existe
if not os.path.exists(CARPETA_DESTINO):
    os.makedirs(CARPETA_DESTINO)
    print(f"Carpeta creada: {CARPETA_DESTINO}")

# Construir ruta completa
ruta_completa = os.path.join(CARPETA_DESTINO, NOMBRE_ARCHIVO)

# Comprobar si existe
if os.path.exists(ruta_completa):
    print(f"\n⚠️  ALERTA:")
    print(f"   El archivo '{NOMBRE_ARCHIVO}' YA EXISTE en '{CARPETA_DESTINO}'.")
    print(f"   ❌ Operación cancelada: No se ha sobreescrito el archivo.")
    print(f"   (Por favor, borra el archivo antiguo o cambia el NOMBRE_ARCHIVO).")
else:
    try:
        with open(ruta_completa, 'w', encoding='utf-8') as f:
            json.dump(interactions_data, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ ¡Éxito! Archivo guardado correctamente en: {ruta_completa}")
        print(f"   Usuarios exportados: {list(interactions_data.keys())}")
        
    except Exception as e:
        print(f"Error al escribir el archivo JSON: {e}")