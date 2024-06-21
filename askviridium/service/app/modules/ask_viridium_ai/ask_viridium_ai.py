import json
import os
from typing import List
import dotenv

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

from ...global_constants import GlobalConstants

dotenv.load_dotenv()

os.environ["OPENAI_API_VERSION"] = "2024-02-01"
model_name = 'gpt-4o'
deployment_name = "langchain-askvai-test-4o"


class ChemicalInfo(BaseModel):
    name: str = Field(description="Name of the chemical")
    cas_no: str = Field(description="CAS number of the chemical")
    source: str = Field(description="Source for this piece of information, will be a hyperlink")


class ChemicalComposition(BaseModel):
    product_name: str = Field(description="Name of the product specified")
    chemicals: List[ChemicalInfo] = Field(description="List of chemicals present in the product.")
    confidence: int = Field(description="Confidence score of the result")


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


class AskViridium:
    def __init__(self):
        self.llm = llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            temperature=0,
            max_tokens=800,
            n=3
        )
        self.cheminfo_prompt = self.prompt1_init()
        self.analysis_prompt = self.prompt2_init()
        self.cheminfo_function, self.analysis_function = self.openai_functions_creation()
        self.cheminfo_model, self.analysis_model = self.bind_function()
        self.parser = JsonOutputFunctionsParser()
        self.cheminfo_chain = self.cheminfo_prompt | self.cheminfo_model | self.parser
        self.analysis_chain = self.analysis_prompt | self.analysis_model | self.parser
        self.chemical_composition = None

        self.result = str()

    def prompt1_init(self):
        with open('system_prompt_templates/findchemicals_prompt.txt', 'r') as file:
            cheminfo_system_prompt = file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", cheminfo_system_prompt),
            ("human", "Material Name: {material}"),
        ])
        return prompt

    def prompt2_init(self):
        with open('system_prompt_templates/newprompt.txt', 'r') as file:
            analysis_system_prompt = file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", analysis_system_prompt),
            ("human", "Material Name: {material}, manufactured by {manufacturer}. CONTEXT: used as {usecase}. Its chemical composition is: {chemical_composition}")
        ])
        return prompt

    def openai_functions_creation(self):
        cheminfo_function = [convert_to_openai_function(ChemicalComposition)]

        # convert MaterialInfo into an OpenAI function to use for function calling.
        analysis_function = [convert_to_openai_function(MaterialInfo)]
        return [cheminfo_function, analysis_function]

    def bind_function(self):
        cheminfo_model = self.llm.bind_functions(
            functions=self.cheminfo_function,
            function_call={"name": "ChemicalComposition"}
        )
        # binding the function to our LLM to enable function calling.
        analysis_model = self.llm.bind_functions(
            functions=self.analysis_function,
            function_call={"name": "MaterialInfo"}
        )
        return [cheminfo_model, analysis_model]

    def query(self, material_name, manufacturer_name, work_content):
        material = material_name
        manufacturer = manufacturer_name
        work_content = work_content

        self.chemical_composition = self.cheminfo_chain.invoke({"material": material, "example": GlobalConstants.chemical_composition_example})
        chemicals_list = [chemical["name"] for chemical in self.chemical_composition["chemicals"]]

        self.result = self.analysis_chain.invoke({"material": material, "manufacturer": manufacturer, "usecase":work_content, "chemical_composition": chemicals_list, "example": GlobalConstants.analysis_example})

        return self.result

    def store(self):
        # saving results
        with open("compound_twostep.json", 'w') as file:
            json.dump(self.result, file)

