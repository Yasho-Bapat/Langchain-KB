import os
from typing import Dict, List
import dotenv

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser


dotenv.load_dotenv()

"""
Start writing langchain implementation for Ask ViridiumAI service.
"""