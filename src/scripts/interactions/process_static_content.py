###########################################################################################
import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
DATA_DIR = os.path.join(project_path, 'data', 'static_content')

# Ruta al JSON de entrada (ajustar si es necesario)
JSON_INPUT_PATH = os.path.join(DATA_DIR, 'raw_data', 'static_content_raw_data.json')

# Carpeta de salida para los .txt
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed_data')

###########################################################################################

# --- HELPERS ---

def load_json(path):
    if not os.path.exists(path):
        logging.error(f"No se encontró el archivo JSON: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_txt(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    logging.info(f" -> Generado: {path}")


def safe_get(d, *keys, default=None):
    """Navega de forma segura por diccionarios anidados."""
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


###########################################################################################

# --- GENERADORES DE CONTENIDO ---

def build_transcript_txt(data):
    """1) URL del video + transcripción."""
    vid_meta = data.get('vid_metadata_col', {})
    url = vid_meta.get('URL', 'N/A')
    title = vid_meta.get('Title', 'N/A')

    lines = []
    lines.append(f"Título: {title}")
    lines.append(f"URL: {url}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("TRANSCRIPCIÓN")
    lines.append("=" * 80)
    lines.append("")

    subtitles = vid_meta.get('Subtitles', [])
    if isinstance(subtitles, list) and subtitles:
        for sub in subtitles:
            lang = sub.get('lang', 'N/A')
            content = sub.get('content', '')
            lines.append(f"--- Idioma: {lang} ---")
            lines.append("")
            lines.append(content)
            lines.append("")
    else:
        lines.append("(No hay transcripción disponible)")

    return "\n".join(lines)


def build_flashcards_txt(data):
    """2) Flashcards."""
    flashcards = safe_get(data, 'reviewed_col', 'Material', 'flashcards', default=[])

    lines = []
    lines.append("FLASHCARDS")
    lines.append("=" * 80)
    lines.append("")

    if not flashcards:
        lines.append("(No hay flashcards disponibles)")
        return "\n".join(lines)

    for i, card in enumerate(flashcards, start=1):
        concept = card.get('concept', 'N/A')
        description = card.get('description', 'N/A')
        start = card.get('start_interval_fixed', 'N/A')
        end = card.get('end_interval_fixed', 'N/A')

        lines.append(f"Flashcard {i}")
        lines.append(f"Concepto: {concept}")
        lines.append(f"Descripción: {description}")
        lines.append(f"Intervalo: {start} - {end}")
        lines.append("-" * 80)
        lines.append("")

    return "\n".join(lines)


def build_mcq_txt(data):
    """3) Multiple Choice Questions."""
    mcqs = safe_get(data, 'reviewed_col', 'Material', 'multiple_choice_questions', default=[])

    lines = []
    lines.append("MULTIPLE CHOICE QUESTIONS")
    lines.append("=" * 80)
    lines.append("")

    if not mcqs:
        lines.append("(No hay preguntas de opción múltiple disponibles)")
        return "\n".join(lines)

    for i, mcq in enumerate(mcqs, start=1):
        question = mcq.get('question', 'N/A')
        options = mcq.get('options', [])
        correct = mcq.get('correct_answer', 'N/A')
        explanation = mcq.get('explanation', 'N/A')
        start = mcq.get('start_interval_fixed', 'N/A')
        end = mcq.get('end_interval_fixed', 'N/A')

        lines.append(f"Pregunta {i}: {question}")
        lines.append("Opciones:")
        for j, opt in enumerate(options, start=1):
            marker = " (CORRECTA)" if opt == correct else ""
            lines.append(f"  {j}. {opt}{marker}")
        lines.append(f"Respuesta correcta: {correct}")
        lines.append(f"Explicación: {explanation}")
        lines.append(f"Intervalo: {start} - {end}")
        lines.append("-" * 80)
        lines.append("")

    return "\n".join(lines)


def build_open_ended_txt(data):
    """4) Open Ended Questions."""
    oeqs = safe_get(data, 'reviewed_col', 'Material', 'open_ended_questions', default=[])

    lines = []
    lines.append("OPEN ENDED QUESTIONS")
    lines.append("=" * 80)
    lines.append("")

    if not oeqs:
        lines.append("(No hay preguntas abiertas disponibles)")
        return "\n".join(lines)

    for i, oeq in enumerate(oeqs, start=1):
        qid = oeq.get('id', f'openended_{i}')
        question = oeq.get('question', 'N/A')
        start = oeq.get('start_interval_fixed', 'N/A')
        end = oeq.get('end_interval_fixed', 'N/A')

        lines.append(f"[{qid}] Pregunta {i}: {question}")
        lines.append(f"Intervalo: {start} - {end}")
        lines.append("-" * 80)
        lines.append("")

    return "\n".join(lines)


def build_evaluation_questions_txt(data):
    """5) Evaluation Questions."""
    eval_data = safe_get(data, 'reviewed_col', 'Material', 'evaluation_questions', default={})
    questions = eval_data.get('questions', []) if isinstance(eval_data, dict) else []
    final_decision = eval_data.get('final_decision', None) if isinstance(eval_data, dict) else None

    lines = []
    lines.append("EVALUATION QUESTIONS")
    lines.append("=" * 80)
    lines.append("")

    if not questions:
        lines.append("(No hay preguntas de evaluación disponibles)")
    else:
        for i, q in enumerate(questions, start=1):
            question = q.get('question', 'N/A')
            expected = q.get('expected_answer', 'N/A')
            explanation = q.get('explanation', 'N/A')
            pass_if = q.get('pass_if', 'N/A')
            fail_if = q.get('fail_if', 'N/A')
            start = q.get('start_interval_fixed', 'N/A')
            end = q.get('end_interval_fixed', 'N/A')

            lines.append(f"Pregunta {i}: {question}")
            lines.append(f"Respuesta esperada: {expected}")
            lines.append(f"Explicación: {explanation}")
            lines.append(f"Criterio de aprobación: {pass_if}")
            lines.append(f"Criterio de reprobación: {fail_if}")
            lines.append(f"Intervalo: {start} - {end}")
            lines.append("-" * 80)
            lines.append("")

    if final_decision is not None:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Decisión final: {final_decision}")

    return "\n".join(lines)


###########################################################################################

# --- MAIN EXECUTION ---

def main():
    logging.info("▶️ GENERANDO ARCHIVOS TXT DESDE static_content_raw_data.json")

    # 1. Cargar JSON
    logging.info(f"STEP 1: Cargando JSON desde {JSON_INPUT_PATH}...")
    data = load_json(JSON_INPUT_PATH)

    # 2. Crear carpeta de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"STEP 2: Carpeta de salida lista: {OUTPUT_DIR}")

    # 3. Generar cada archivo
    logging.info("STEP 3: Generando archivos...")

    outputs = {
        '1_transcripcion.txt': build_transcript_txt(data),
        '2_flashcards.txt': build_flashcards_txt(data),
        '3_multiple_choice_questions.txt': build_mcq_txt(data),
        '4_open_ended_questions.txt': build_open_ended_txt(data),
        '5_evaluation_questions.txt': build_evaluation_questions_txt(data),
    }

    for filename, content in outputs.items():
        write_txt(os.path.join(OUTPUT_DIR, filename), content)

    logging.info("✅ GENERACIÓN DE ARCHIVOS TXT FINALIZADA CON ÉXITO")


if __name__ == "__main__":
    main()