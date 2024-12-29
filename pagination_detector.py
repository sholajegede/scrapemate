# pagination_detector.py

import os
import json
from typing import List, Dict, Tuple, Union
from pydantic import BaseModel, Field, ValidationError

import tiktoken
from dotenv import load_dotenv


from openai import OpenAI
import google.generativeai as genai
from groq import Groq

import openai
from api_management import get_api_key
from assets import PROMPT_PAGINATION, PRICING, GROQ_LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME_OTHER

load_dotenv()
import logging

class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list, description="List of pagination URLs, including 'Next' button URL if present")

def calculate_pagination_price(token_counts: Dict[str, int], model: str) -> float:
    """
    Calculate the price for pagination based on token counts and the selected model.
    
    Args:
    token_counts (Dict[str, int]): A dictionary containing 'input_tokens' and 'output_tokens'.
    model (str): The name of the selected model.

    Returns:
    float: The total price for the pagination operation.
    """
    input_tokens = token_counts['input_tokens']
    output_tokens = token_counts['output_tokens']
    
    input_price = input_tokens * PRICING[model]['input']
    output_price = output_tokens * PRICING[model]['output']
    
    return input_price + output_price

def detect_pagination_elements(url: str, indications: str, selected_model: str, markdown_content: str) -> Tuple[Union[PaginationData, Dict, str], Dict, float]:
    try:
        """
        Uses AI models to analyze markdown content and extract pagination elements.

        Args:
            selected_model (str): The name of the OpenAI model to use.
            markdown_content (str): The markdown content to analyze.

        Returns:
            Tuple[PaginationData, Dict, float]: Parsed pagination data, token counts, and pagination price.
        """ 
        prompt_pagination = PROMPT_PAGINATION+"\n The url of the page to extract pagination from   "+url+"if the urls that you find are not complete combine them intelligently in a way that fit the pattern **ALWAYS GIVE A FULL URL**"
        if indications != "":
            prompt_pagination +=PROMPT_PAGINATION+"\n\n these are the users indications that, pay special attention to them: "+indications+"\n\n below are the markdowns of the website: \n\n"
        else:
            prompt_pagination +=PROMPT_PAGINATION+"\n There are no user indications in this case just apply the logic described. \n\n below are the markdowns of the website: \n\n"

        if selected_model == "Groq Llama3.1 70b":
            # Use Groq client
            client = Groq(api_key=get_api_key("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_LLAMA_MODEL_FULLNAME,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
            )
            response_content = response.choices[0].message.content.strip()
            # Try to parse the JSON
            try:
                pagination_data = json.loads(response_content)
            except json.JSONDecodeError:
                pagination_data = {"page_urls": []}
            # Token counts
            token_counts = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            # Calculate the price
            pagination_price = calculate_pagination_price(token_counts, selected_model)

            '''# Ensure the pagination_data is a dictionary
            if isinstance(pagination_data, PaginationData):
                pagination_data = pagination_data.model_dump()
            elif not isinstance(pagination_data, dict):
                pagination_data = {"page_urls": []}'''

            return pagination_data, token_counts, pagination_price

        elif selected_model == "Groq Llama3.3 70b":
            # Use Groq client
            client = Groq(api_key=get_api_key("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_LLAMA_MODEL_FULLNAME_OTHER,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
            )
            response_content = response.choices[0].message.content.strip()
            # Try to parse the JSON
            try:
                pagination_data = json.loads(response_content)
            except json.JSONDecodeError:
                pagination_data = {"page_urls": []}
            # Token counts
            token_counts = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            # Calculate the price
            pagination_price = calculate_pagination_price(token_counts, selected_model)

            '''# Ensure the pagination_data is a dictionary
            if isinstance(pagination_data, PaginationData):
                pagination_data = pagination_data.model_dump()
            elif not isinstance(pagination_data, dict):
                pagination_data = {"page_urls": []}'''

            return pagination_data, token_counts, pagination_price

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        logging.error(f"An error occurred in detect_pagination_elements: {e}")
        # Return default values if an error occurs
        return PaginationData(page_urls=[]), {"input_tokens": 0, "output_tokens": 0}, 0.0