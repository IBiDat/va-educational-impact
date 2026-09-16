from itertools import islice
import os 
import json
from typing import TypedDict

from google import genai
from google.genai import types
from tqdm import tqdm

import polars as pl

class PromptTypeJSON(TypedDict):
    classification: str
    reasoning: str

#########################################################################################################################################################

def return_llm_interaction_type(
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
        response_schema=PromptTypeJSON
      )
    )
    
    response = json.loads(response.text)
    
    return response

#########################################################################################################################################################

def return_interaction_type(
    raw_data_path: str,
    template_path: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    #Read raw_data_path
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data: dict[str, dict[str, list|dict]] = json.load(f)
    
    #Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    #Create LLM Client
    client = genai.Client()
    
    cheating_data = []
    for user, chat_interactions_list in tqdm(raw_data.items()):
        for i, interaction in enumerate(chat_interactions_list):
            user_input = interaction["user_input"]
            semantic_depth_level = interaction["semantic_depth_level"]
            
            if semantic_depth_level < 2:
                #Skip off-topic and cheating interactions
                continue
            
            #Get LLM Interaction Type
            llm_answer = return_llm_interaction_type(
                client=client,
                prompt=template.format(
                    USER_INTERACTION=user_input
                )
            )

            cheating_data.append([
                f"{user}_{i}",
                user,
                user_input,
                llm_answer["classification"],
                llm_answer["reasoning"]
            ])

    
    #Create cheating df
    interaction_type_df = pl.DataFrame(
        data=cheating_data,
        schema={
            "conversation_id": pl.String,
            "id": pl.String,
            "user_input": pl.String,
            "prompt_type": pl.String,
            "reasoning_llm": pl.String
        },
        orient="row"
    )
    
    #Group results by users
    grouped_interaction_type_df = interaction_type_df.pivot(
        on="prompt_type",
        index="id",
        values="prompt_type",
        aggregate_function="len"
    ).fill_null(0)
    
    return interaction_type_df, grouped_interaction_type_df
    
    