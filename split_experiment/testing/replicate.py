import json
import os
from typing import List
import dotenv

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

dotenv.load_dotenv()

os.environ["OPENAI_API_VERSION"] = "2024-02-01"
# model_name = 'gpt-3.5-turbo'
# deployment_name = "langchain-splitting-test"
model_name = 'gpt-4o'
deployment_name = "langchain-askvai-test-4o"


# Structure of the desired output. This is passed to the LLM as an instruction on how to return the response to query.
class MaterialInfo(BaseModel):
    """Information to extract."""
    analyzed_material: str = Field(description="Name of the material that was analyzed")
    composition: str = Field(description="Composition of the material")
    analysis_method: str = Field(description="How its PFAS analysis was conducted - methods, sources")
    decision: str = Field(description="Decision of whether the material is PFAS compliant or not: PFAS (Yes/No)")
    confidence: float = Field(description="Confidence score of response.")
    primary_reason: str = Field(description="Primary reasoning of response content.")
    secondary_reason: str = Field(description="Secondary reasoning of response content.")
    evidence: List[str] = Field(description="Evidence supporting the response given.")
    confidence_level: str = Field(description="Confidence level of response between low, medium and high.")
    recommendation: str = Field(description="Recommendation of what to do with the material with regards to its PFAS "
                                            "compliance.")
    suggestion: str = Field(description="Suggestion of what to do with the material with regards to its PFAS "
                                        "compliance.")
    limitations_and_uncertainties: str = Field(description="Limitations and uncertainties of material and its PFAS "
                                                           "compliance based on the data that could be looked up.")


# An example of what we want. This is passed during the system message prompt to the LLM. Based on MaterialInfo class.
example = {
    "analyzed_material": "0652-W Nylon/ 30655-W nylon with CPT Sealant",
    "composition": "Nylon, CPT Sealant",
    "analysis_method": "Literature review, trade name association",
    "decision": "PFAS (No)",
    "confidence_score": 0.90,
    "primary_reason": "Nylon is a polymer that does not contain PFAS. CPT Sealant does not typically contain PFAS based on available information.",
    "secondary_reason": None,
    "evidence": ["Trade name association with nylon, which is a non-PFAS material",
                 "Lack of information suggesting PFAS presence in CPT Sealant"],
    "confidence_level": "High",
    "recommendation": "No further investigation is needed as the analyzed materials are not expected to contain PFAS.",
    "suggestion": None,
    "limitations_and_uncertainties": None
}

# reading the system message from an external file (easy to track).
with open('../../askviridium/ask_viridium_ai/system_prompt_templates/newprompt.txt', 'r') as file:
    template = file.read()

# Initialize output parser.
parser = JsonOutputFunctionsParser()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", template),
#     ("human", "Material Name: {material}, Manufactured by {manufacturer}, used as {work_content}."),
# ])
prompt_s = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "Material Name: {material}"),
])

prompt_c = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "Material Name: {material}, manufactured by {manufacturer}, used as {usecase}"),
])
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    #model_name=model_name,
    temperature=0,
    max_tokens=800,
    n=2
)

# convert MaterialInfo into an OpenAI function to use for function calling.
aoai_function = [
    convert_to_openai_function(MaterialInfo)
]

# binding the function to our LLM to enable function calling.
model = llm.bind_functions(
    functions=aoai_function,
    function_call={"name": "MaterialInfo"}
)

# creating the chain prompt -> LLM -> JSONOutputParser
chain_s = prompt_s | model | parser  # simple chain
chain_c = prompt_c | model | parser  # compound chain

material = "Blasocut 4000"
manufacturer = "Blaser Swisslube, Inc."
work_content = 'Coolant'

result_c = chain_c.invoke(
    {"material": material, "manufacturer": manufacturer, "usecase": work_content, "example": example})

# call the LLM, and send material information and our example. {example} is a field in the template.
result_s = chain_s.invoke({"material": material, "example": example})
with open("simple.json", 'w') as file:
    json.dump(result_s, file)

with open("compound.json", 'w') as file:
    json.dump(result_c, file)

print(result_c)
print(result_s)
