import os 
import re
import json
from typing import TypedDict

from google import genai
from google.genai import types
from tqdm import tqdm

import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CheatingLLMScoreOutput(TypedDict):
    label: int
    reasoning: str

#########################################################################################################################################################

def process_forms_questions(
    questions: list[str]
) -> list[str]:
    
    #Exclude 'Marca temporal', 'Puntuación', 'IDENTIFICADOR'
    questions = [
        q
        for q in questions
        if q not in ['Marca temporal', 'Puntuación', 'IDENTIFICADOR']
    ]
    
    #Remove [] elements
    pattern = r'\[.*?\]'
    clean_questions = [re.sub(pattern, '', pregunta).strip() for pregunta in questions]
    
    return clean_questions

#########################################################################################################################################################

def get_forms_questions(forms_dir: str) -> list[str]:
    pre_questions = pl.read_csv(
        os.path.join(forms_dir, "respuestas_forms_pre.csv")
    ).columns
    
    post_questions = pl.read_csv(
        os.path.join(forms_dir, "respuestas_forms_post.csv")
    ).columns
    
    clean_pre_questions = process_forms_questions(
        questions=pre_questions
    )
    clean_post_questions = process_forms_questions(
        questions=post_questions
    )
    
    return clean_pre_questions + clean_post_questions

#########################################################################################################################################################

def get_static_content_questions(static_content_path: str) -> list[str]:
    """
    Parses a structured text string to extract question numbers and their content.
    """
    # Load static content
    with open(static_content_path, "r", encoding="utf-8") as f:
        static_content_dict = json.load(f)
    
    # Get questions
    questions = []
    #---------------------------------------------------------------
    # Multiple choice questions
    multiple_choice_questions = static_content_dict["reviewed_col"]["Material"]["multiple_choice_questions"]
    for q in multiple_choice_questions:
        questions.append(q["question"])
        answers = " ".join(q["options"])   
        questions.append(q["question"] + answers)
    
    # Open-ended questions
    open_ended_questions = static_content_dict["reviewed_col"]["Material"]["open_ended_questions"]
    for q in open_ended_questions:
        questions.append(q["question"])
        
    # Evaluation questions
    evaluation_questions = static_content_dict["reviewed_col"]["Material"]["evaluation_questions"]["questions"]
    for q in evaluation_questions:
        questions.append(q["question"])
    
    return questions

#########################################################################################################################################################

def return_llm_cheating_score(
    client: str,
    prompt: str
) -> dict[str, str|int]:
    
    #Generate model answer
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=CheatingLLMScoreOutput
      )
    )
    
    response = json.loads(response.text)
    
    return response

#########################################################################################################################################################

def compute_cheating_score(
    raw_data_path: str,
    template_path: str,
    every_question: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    #Read raw_data_path
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data: dict[str, dict[str, list|dict]] = json.load(f)
    
    #Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    #Create LLM Client
    client = genai.Client()
    
    # Load a pre-trained sentence encoder model
    model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')
    questions_embeddings = model.encode(every_question)
    
    cheating_data = []
    for user, interactions in tqdm(raw_data.items()):
        chat_interactions_list = interactions["chat_interactions"]
        if len(chat_interactions_list) == 0:
            continue
        else:
            for i, chat_interaction in enumerate(chat_interactions_list):
                user_input = chat_interaction["user_input"]
                user_input_embedding = model.encode(user_input)
                
                # Compute cosine similarity
                similarities = cosine_similarity(
                    [user_input_embedding],
                    questions_embeddings
                )[0]

                # Find the maximum similarity and the corresponding phrase
                max_similarity = np.max(similarities)
                max_index = np.argmax(similarities)
                most_similar_question = every_question[max_index]
                
                #Compute cheating score: Similarity > 0.85
                cheating_score_encoder = 0
                if max_similarity > 0.85: cheating_score_encoder = 1
                
                #Get LLM Cheating Score
                llm_answer = return_llm_cheating_score(
                    client=client,
                    prompt=template.format(
                        QUESTION_LIST=every_question,
                        USER_INTERACTION=user_input
                    )
                )

                cheating_data.append([
                    f"{user}_{i}",
                    user,
                    user_input,
                    most_similar_question,
                    max_similarity,
                    cheating_score_encoder,
                    llm_answer["label"],
                    llm_answer["reasoning"]
                ])
    
    #Create cheating df
    cheating_df = pl.DataFrame(
        data=cheating_data,
        schema={
            "conversation_id": pl.String,
            "id": pl.String,
            "user_input": pl.String,
            "most_similar_question": pl.String,
            "similarity_score": pl.Float64,
            "cheating_score_encoder": pl.UInt64,
            "cheating_score_llm": pl.UInt64,
            "reasoning_llm": pl.String
        },
        orient="row"
    )
    
    #Group results by users
    grouped_cheating_df = cheating_df.select(
        "id",
        "cheating_score_encoder",
        "cheating_score_llm"
    ).group_by(
        "id"
    ).agg(
        cheating_score_encoder = pl.col("cheating_score_encoder").sum()/pl.col("id").len(),
        cheating_score_llm = pl.col("cheating_score_llm").sum()/pl.col("id").len()
    )
    
    return cheating_df, grouped_cheating_df
    
    