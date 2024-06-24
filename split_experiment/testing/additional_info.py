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
PROMPT_COST_PER_1000_TOKENS = 0.03
COMPLETION_COST_PER_1000_TOKENS = 0.06
CONFIDENCE_THRESHOLD = 0.91


# Function to add spacing between print statements
def spacing(spacing_level: str = "medium"):
    gap = "-----------------------------------------------" * 3
    if spacing_level == "large":
        print(f"\n\n{gap}\n")
    elif spacing_level == "medium":
        print(f"\n{gap}\n")
    elif spacing_level == "small":
        print("\n\n")


# Function to check if additional information is needed
def needs_additional_info(result, i):
    variant = result["variants"][i]
    if variant["decision"].lower() in ["undetermined", "pfas (undetermined)"] or variant["confidence"] < CONFIDENCE_THRESHOLD:
        return True
    return False


# Function to count tokens and calculate cost
def count_tokens_and_cost(prompt_tokens, completion_tokens):
    total_tokens = prompt_tokens + completion_tokens
    prompt_cost = (prompt_tokens / 1000) * PROMPT_COST_PER_1000_TOKENS
    completion_cost = (completion_tokens / 1000) * COMPLETION_COST_PER_1000_TOKENS
    total_cost = prompt_cost + completion_cost
    return total_tokens, total_cost


class MaterialInfo(BaseModel):
    """Information to extract."""
    analyzed_material: str = Field(description="Name of the material that was analyzed")
    type_of_material: str = Field(description="Name of actual material - ex - defoamant, reamer, etc.")
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
    variants: List[MaterialInfo] = Field(description="List of TWO variants of analysis (try different inputs).")


class ChemicalInfo(BaseModel):
    name: str = Field(description="Name of the chemical")
    cas_no: str = Field(description="CAS number of the chemical")
    source: str = Field(description="Source for this piece of information, will be a hyperlink")


class ChemicalComposition(BaseModel):
    product_name: str = Field(description="Name of the product specified")
    chemicals: List[ChemicalInfo] = Field(description="List of chemicals present in the product.")
    confidence: int = Field(description="Confidence score of the result")


print(f"Initializing Prompt\nEstablishing Connection with LLM({model_name})...")
spacing("medium")

chemical_composition_example = {
    "product_name": "TRIM TC 184B",
    "chemicals": [{"name": "Severely Hydrotreated Petroleum Oil", "cas_no": "64742-65-0",
                   "source": "https://www1.mscdirect.com/MSDS/MSDS00007/01790583-20110708.PDF"}],
    "confidence": 0.85
}

analysis_example = {
    "analyzed_material": "0652-W Nylon/ 30655-W nylon with CPT Sealant",
    "composition": "Nylon, CPT Sealant",
    "analysis_method": "Literature review, trade name association",
    "decision": "PFAS (No)",
    "confidence_score": 0.40,
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

# Initialize output parser.
parser = JsonOutputFunctionsParser()

# Initialize AzureChatOpenAI model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    model_name=model_name,
    temperature=0.18,
    max_tokens=800,
)

with open('../../askviridium/ask_viridium_ai/findchemicals_prompt.txt', 'r') as file:
    chemical_template = file.read()

chemical_prompt = ChatPromptTemplate.from_messages([
    ("system", chemical_template),
    ("human", "Material Name: {material}"),
])

chemical_func = [
    convert_to_openai_function(ChemicalComposition)
]

chemical_model = llm.bind(
    functions=chemical_func,
    function_call={"name": "ChemicalComposition"}
)

chemical_chain = chemical_prompt | chemical_model | parser

with open('../../askviridium/ask_viridium_ai/system_prompt_templates/prompt_using_gemini.txt', 'r') as file:
    analysis_template = file.read()

# Define the prompt template
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", analysis_template),
    ("human",
     "Material Name: {material}, Manufactured by {manufacturer}, used as {work_content} with the follwoing chemical composition: {chemical_composition}. Additional information:{additional_info}"),
])

analysis_func = [
    convert_to_openai_function(Results)
]

# Bind the function to the LLM to enable function calling.
analysis_model = llm.bind(
    functions=analysis_func,
    function_call={"name": "Results"}
)

# Create the chain: prompt -> model -> parser
analysis_chain = analysis_prompt | analysis_model | parser

material = "25.1 SDS Sanodure Grey HLN Liq (Aluminum)"
manufacturer = "Clariant Plastics & Coating USA Inc."
work_content = "Not Available"

print(f"Material Name: {material}")
print(f"Manufactured by: {manufacturer}")
print(f"Used as: {work_content}")
spacing("medium")

composition = chemical_chain.invoke({"material": material, "example": chemical_composition_example})

print("Generating Response...")
spacing("medium")

print("Composition Analysis:\n")
for chemical in composition["chemicals"]:
    print(f"{chemical["name"]}: {chemical['cas_no']}  ({chemical['source']})")

result = analysis_chain.invoke(
    {"material": material, "manufacturer": manufacturer, "work_content": work_content,
     "chemical_composition": composition, "example": analysis_example, "additional_info": None})

spacing("medium")
print("Extracted Entities: \n")

for variant in result["variants"]:
    print(f"Type of Material: {variant["type_of_material"]}")
    print(f"Composition: {variant['composition']}")
    print(f"Analysis Method: {variant['analysis_method']}")
    print(f"Decision: {variant['decision']}")
    print(f"Confidence Score: {variant['confidence']}")
    print(f"Primary Reason: {variant['primary_reason']}")
    print(f"Secondary Reason: {variant['secondary_reason']}")
    print(f"Evidence: {variant['evidence']}")
    print(f"Confidence Level: {variant['confidence_level']}")
    print(f"Recommendation: {variant['recommendation']}")
    print(f"Suggestion: {variant['suggestion']}")
    print(f"Limitations and Uncertainties: {variant['limitations_and_uncertainties']}")
    spacing("large")

if len(result["variants"]) == 2:
    ask = input("Which response was better - 1 or 2? [1 (default) / 2]: ")
    better_variant = 1 if ask == "" else int(ask)
    print(f"\n{better_variant} was better, noted!.")
    spacing("large")
else:
    better_variant = 1

if needs_additional_info(result, better_variant - 1):
    spacing("small")
    print("Additional information is required based on the analysis:")
    print(f"Material Name: {material}")
    print(f"Decision: {result["variants"][better_variant - 1]['decision']}")
    print(f"Confidence Score: {result["variants"][better_variant - 1]['confidence']}")
    print(f"Primary Reason: {result["variants"][better_variant - 1]['primary_reason']}")
    print("Please provide additional information to improve the analysis.")

    spacing("medium")
    type_of_material = input(
        f"Would you like me to explore other {result["variants"][better_variant - 1]['type_of_material']}s of the same composition and use that infer the potential presence of PFAS in the original material? [y/n] ")

    additional_info = str()
    if type_of_material.lower() == "y" or type_of_material.lower() == "":
        additional_info = f"Pls explore other {result["variants"][better_variant - 1]['type_of_material']}s of the same composition and use that infer the potential presence of PFAS in the original material."

    spacing("small")
    additional_info += input("Additional Information: ")

    updated_input = {
        "material": material,
        "manufacturer": manufacturer,
        "work_content": work_content,
        "chemical_composition": composition,
        "example": analysis_example,
        "additional_info": additional_info + ". Also try not to give Undetermined PFAS status this time but don't "
                                             "give false results."
    }

    result = analysis_chain.invoke(updated_input)

    spacing("large")
    print(result)
