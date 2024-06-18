import os
from typing import Dict
import dotenv

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureChatOpenAI

dotenv.load_dotenv()


class DataExtract(BaseModel):
    """Information to extract."""
    chemicals: Dict[str, str] = Field(description="chemical names as keys and CAS numbers as values")
    trade_name: str = Field(description="The trade name of the product.")
    manufacturer: str = Field(description="The name of the manufacturer.")
    hazard_info: str = Field(description="Information about hazards associated with the product/chemicals.")


os.environ["OPENAI_API_VERSION"] = "2024-02-01"
model_name = 'gpt-3.5-turbo'

deployment_name = "langchain-splitting-test"

template = f"""You are an expert at extracting data from given text.
Don't try to make up or guess the answer if you don't know the answer.
Do not output any javascript code at any cost, OUTPUT FORMAT MUST BE JSON. The JSON must have a dictionary containing:
- "chemicals": a dictionary with chemical names as keys and their CAS numbers as values.
- "manufacturer": the name of the manufacturer.
- "trade_name": the trade name of the product.
- "hazard_info": the hazard information associated with the product.
DO NOT OUTPUT ANY JAVASCRIPT CODE.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}"),
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

dataextract_chain = prompt | dataextract_model

file_path = "../docs/8.1 SDS COPPER SULFATE.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
print(" ".join([document.page_content for document in documents]))
print(f"Loaded {len(documents)} documents from {file_path}")

result = dataextract_chain.invoke({"input": documents[0].page_content})
dataextract_json = result.additional_kwargs["function_call"]["arguments"]

print(dataextract_json)
