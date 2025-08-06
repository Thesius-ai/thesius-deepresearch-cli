from typing import Literal, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
import json
from dotenv import load_dotenv

from .struct import SchemaField

load_dotenv()

def init_llm(
        provider: Literal["openai", "anthropic", "google", "ollama"],
        model: str,
        temperature: float = 0.5,
):
    """
    Initialize and return a language model chat interface based on the specified provider.

    This function creates a chat interface for different LLM providers including OpenAI, 
    Anthropic, Google, and Ollama. It handles API key validation and configuration for
    each provider.

    Args:
        provider: The LLM provider to use. Must be one of "openai", "anthropic", "google", or "ollama".
        model: The specific model name/identifier to use with the chosen provider.
        temperature: Controls randomness in the model's output. Higher values (e.g. 0.8) make the output
                    more random, while lower values (e.g. 0.2) make it more deterministic. Defaults to 0.5.

    Returns:
        A configured chat interface for the specified provider and model.

    Raises:
        ValueError: If the required API key environment variable is not set for the chosen provider
                   (except for Ollama which runs locally).
    """
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please set it in your environment variables.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "google":
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=os.environ["GOOGLE_API_KEY"])
    elif provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)



def process_datagen_prompt(fields: List[SchemaField], rows: int = 10) -> str:
    schema_instruction = {field.key: field.description for field in fields}

    field_string = f"""## Response Format
Always respond with a valid JSON array of objects:
[
{json.dumps(schema_instruction, indent=2)},
// Additional entries...
]
"""
    return f"""
You are an expert Question-Answer generation assistant who has the skills of a polymath. Your task is to analyze content provided by the user and generate a comprehensive set of questions with detailed answers based on that content.

## Core Instructions

1. When presented with content, carefully analyze it to identify key concepts, important details, practical applications, and potential challenges or edge cases.

2. Generate a diverse set of questions and answers that thoroughly cover the provided content. Your response must be in valid JSON format.

3. Format code properly within JSON strings, using appropriate escape characters for special characters.

4. Number of dataset rows must be {rows}

{field_string}
"""