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

class ChemicalInfo(BaseModel):
    name: str = Field(description="Name of the chemical")
    cas_no: str = Field(description="CAS number of the chemical")
    source: str = Field(description="Source for this piece of information, will be a hyperlink")
class ChemicalComposition(BaseModel):
    product_name: str = Field(description="Name of the product specified")
    chemicals: List[ChemicalInfo] = Field(description="List of chemicals present in the product.")
    confidence: int = Field(description="Confidence score of the result")

chemical_composition_example = {
    "product_name": "TRIM TC 184B",
    "chemicals": [{"name": "Severely Hydrotreated Petroleum Oil", "cas_no": "64742-65-0", "source": "https://www1.mscdirect.com/MSDS/MSDS00007/01790583-20110708.PDF"}],
    "confidence": 0.85
}
class MaterialInfo(BaseModel):
    """Information to extract."""
    analyzed_material: str = Field(description="Name of the material that was analyzed")
    composition: str = Field(description="Chemical composition of the material")
    analysis_method: str = Field(description="How its PFAS analysis was conducted - methods, sources")
    decision: str = Field(description="Decision of whether the material is PFAS compliant or not: PFAS (Yes/No)")
    confidence: float = Field(description="Confidence score of response.")
    primary_reason: str = Field(description="Primary reasoning of response content.")
    secondary_reason: str = Field(description="Secondary reasoning of response content.")
    evidence: List[str] = Field(description="Evidence supporting the response given.")
    health_problems: List[str] = Field(description="List of health problems that could potentially be attached to the product.")
    confidence_level: str = Field(description="Confidence level of response between low, medium and high.")
    recommendation: str = Field(description="Recommendation of what to do with the material with regards to its PFAS compliance.")
    suggestion: str = Field(description="Suggestion of what to do with the material with regards to its PFAS compliance.")
    limitations_and_uncertainties: str = Field(description="Limitations and uncertainties of material and its PFAS compliance based on the data that could be looked up.")


# An example of what we want. This is passed during the system message prompt to the LLM. Based on MaterialInfo class.
analysis_example = {
"analyzed_material": "0652-W Nylon/ 30655-W nylon with CPT Sealant",
"composition": "Nylon, CPT Sealant",
"analysis_method": "Literature review, trade name association",
"decision": "PFAS (No)",
"confidence_score": 0.90,
"primary_reason": "Nylon is a polymer that does not contain PFAS. CPT Sealant does not typically contain PFAS based on available information.",
"secondary_reason": None,
"evidence": ["Trade name association with nylon, which is a non-PFAS material", "Lack of information suggesting PFAS presence in CPT Sealant"],
"health_problems": ["Could lead to asphyxia", "Linked to cancer"],
"confidence_level": "High",
"recommendation": "No further investigation is needed as the analyzed materials are not expected to contain PFAS.",
"suggestion": None,
"limitations_and_uncertainties": None
}


# Initialize LLM
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    temperature=0,
    max_tokens=800,
    n=3
)


with open('system_prompt_templates/findchemicals_prompt.txt', 'r') as file:
    cheminfo_system_prompt = file.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", cheminfo_system_prompt),
    ("human", "Material Name: {material}"),
])


# reading the system message from an external file (easy to track).
with open('system_prompt_templates/newprompt.txt', 'r') as file:
    analysis_system_prompt = file.read()


prompt_s = ChatPromptTemplate.from_messages([
    ("system", analysis_system_prompt),
    ("human", "Material Name: {material}"),
])

prompt_c = ChatPromptTemplate.from_messages([
    ("system", analysis_system_prompt),
    ("human", "Material Name: {material}, manufactured by {manufacturer}. CONTEXT: used as {usecase}. Its chemical composition is: {chemical_composition}")
])

cheminfo_function = [convert_to_openai_function(ChemicalComposition)]

# convert MaterialInfo into an OpenAI function to use for function calling.
analysis_function = [
    convert_to_openai_function(MaterialInfo)
]

cheminfo_model = llm.bind_functions(
    functions=cheminfo_function,
    function_call={"name": "ChemicalComposition"}
)
# binding the function to our LLM to enable function calling.
analysis_model = llm.bind_functions(
    functions=analysis_function,
    function_call={"name": "MaterialInfo"}
)

# Initialize output parser.
parser = JsonOutputFunctionsParser()

# creating the chain prompt -> LLM -> JSONOutputParser
chain_cheminfo = prompt | cheminfo_model | parser
chain_s = prompt_s | analysis_model | parser # simple chain
chain_c = prompt_c | analysis_model | parser # compound chain

material = "Blasocut 4000"
manufacturer = "Blaser Swisslube, Inc."
work_content = 'Coolant'

chemical_composition = chain_cheminfo.invoke({"material": material, "example": chemical_composition_example})

print(chemical_composition)
chemicals_list = [chemical["name"] for chemical in chemical_composition["chemicals"]]

result_s = chain_s.invoke({"material": material, "example": analysis_example})
result_c = chain_c.invoke({"material": material, "manufacturer": manufacturer, "usecase":work_content, "chemical_composition": chemicals_list, "example": analysis_example})


# saving results
with open ("simple_twostep.json", 'w') as file:
    json.dump(result_s, file)

with open("compound_twostep.json", 'w') as file:
    json.dump(result_c, file)


#printing results
print()
print("SIMPLE")
print(result_s)
print()
print("COMPOUND")
print(result_c)

