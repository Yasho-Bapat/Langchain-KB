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
model_name = 'gpt-4o'
deployment_name = "langchain-askvai-test-4o"


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
    recommendation: str = Field(
        description="Recommendation of what to do with the material with regards to its PFAS compliance.")
    suggestion: str = Field(
        description="Suggestion of what to do with the material with regards to its PFAS compliance.")
    limitations_and_uncertainties: str = Field(
        description="Limitations and uncertainties of material and its PFAS compliance based on the data that could be looked up.")


class Results(BaseModel):
    variants: List[MaterialInfo] = Field(description="List of 2 variants of analysis (try different inputs).")


example = {
    "analyzed_material": "0652-W Nylon/ 30655-W nylon with CPT Sealant",
    "composition": "Nylon, CPT Sealant",
    "analysis_method": "Literature review, trade name association",
    "decision": "PFAS (No)",
    "confidence_score": 0.90,
    "primary_reason": "Nylon is a polymer that does not contain PFAS. CPT Sealant does not typically contain PFAS "
                      "based on available information.",
    "secondary_reason": None,
    "evidence": ["Trade name association with nylon, which is a non-PFAS material",
                 "Lack of information suggesting PFAS presence in CPT Sealant"],
    "confidence_level": "High",
    "recommendation": "No further investigation is needed as the analyzed materials are not expected to contain PFAS.",
    "suggestion": None,
    "limitations_and_uncertainties": None
}

# reading the system message from an external file (easy to track).
with open('newprompt2.txt', 'r') as file:
    template = file.read()

# Initialize output parser.
parser = JsonOutputFunctionsParser()

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "Material Name: {material}, Manufactured by {manufacturer}, used as {work_content}. Additional information:{additional_info}"),
])

# Initialize AzureChatOpenAI model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    model_name=model_name,
    temperature=0.18,
    max_tokens=800,
)

# Convert Results into an OpenAI function for function calling.
aoai_function = [
    convert_to_openai_function(Results)
]

# Bind the function to the LLM to enable function calling.
model = llm.bind(
    functions=aoai_function,
    function_call={"name": "Results"}
)

# Create the chain: prompt -> model -> parser
chain = prompt | model | parser


# Function to check if additional information is needed
def needs_additional_info(result, i):
    variant = result["variants"][i]
    if variant["decision"].lower() in ["undetermined", "pfas (undetermined)"] or variant["confidence"] < 0.6:
        return True
    return False


material = 'Resin Bonded WHEEL 69078666546'
manufacturer = 'Master Fluid Solutions'
work_content = 'Coolant'

result = chain.invoke(
    {"material": material, "manufacturer": manufacturer, "work_content": work_content, "example": example, "additional_info": None})

print(result)

if len(result["variants"]) == 2:
    better_variant = int(input("Which response was better - 1 or 2? [1 (default) / 2]: "))
    better_variant = 1 if better_variant == "" else better_variant
else:
    better_variant = 1

if needs_additional_info(result, better_variant - 1):
    print("Additional information is required based on the analysis:")
    print(f"Material Name: {material}")
    print(f"Decision: {result["variants"][better_variant - 1]['decision']}")
    print(f"Confidence Score: {result["variants"][better_variant - 1]['confidence']}")
    print(f"Primary Reason: {result["variants"][better_variant - 1]['primary_reason']}")
    print("Please provide additional information to improve the analysis.")
    additional_info = input("Additional Information: ")

    updated_input = {
        "material": material,
        "manufacturer": manufacturer,
        "work_content": work_content,
        "example": example,
        "additional_info": additional_info
    }

    result = chain.invoke(updated_input)

    print(result)
