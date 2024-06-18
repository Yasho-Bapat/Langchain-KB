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

class Chemicals(BaseModel):
    name: str
    cas_numer: str

class ManufacturerInfo(BaseModel):
    name: str
    address: str
class DataExtract(BaseModel):
    """Information to extract."""
    chemicals: Chemicals
    # chemicals: List[str] = Field(description="names of chemicals mentioned in the document")
    product_name: str = Field(description="The trade name of the product.")
    manufacturer: ManufacturerInfo
    hazard_info: str = Field(description="Information about hazards associated with the product/chemicals.")

example = {
    "chemicals": {"name": "Acetone", "CAS_number": "67-64-1"},
    "product_name": "product ABC",
    "manufacturer_info": {"name": "ABC Corp. Ltd.", "address": "1, Street ABC, City DEF, CA 49023"},
    "hazard_info": "highly flammable"
}

os.environ["OPENAI_API_VERSION"] = "2024-02-01"
model_name = 'gpt-3.5-turbo'

deployment_name = "langchain-splitting-test"

template = """You are an expert at extracting data from given text.
Don't try to make up or guess the answer if you don't know the answer.\n
Always return chemical names and their CAS numbers listed in the document, along with what the query is requesting.
Response should be a JSON object.
DO NOT RETURN ANY CODE. ONLY RETURN THE JSON OBJECT. Here is an example of how an output should be: \n {example}
"""

parser = JsonOutputFunctionsParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "Context: {context}\n Query: {query}"),
])

llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    model_name=model_name,
    temperature=0,
    max_tokens=800,
)


dataextract_function = [
    convert_to_openai_function(DataExtract)
]

dataextract_model = llm.bind(
    functions=dataextract_function,
    function_call={"name": "DataExtract"}
)

dataextract_chain = prompt | dataextract_model | parser

file_path = "../docs/1.1 SDS Havaklean KP.PDF"
loader = PyPDFLoader(file_path)
documents = loader.load()
print(f"Loaded {len(documents)} documents from {file_path}")
#print(documents[2].page_content)

# NOT PASSING FULL CONTEXT HERE DUE TO TOKEN LIMIT. IDEALLY, THE FINAL JSON BODY NEEDS TO BE EDITED WITH INFORMATION FROM ALL PAGES IN THE DOCUMENT.
result = dataextract_chain.invoke({"context": documents[0].page_content, "query": "Return information about the contents of the document. Give chemical names and CAS numbers as well.", "example": example})

print(result)
