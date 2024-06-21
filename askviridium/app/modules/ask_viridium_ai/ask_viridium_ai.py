import dotenv
import json

from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

from askviridium.app.global_constants import GlobalConstants
from askviridium.app.models import ChemicalComposition, MaterialInfo

dotenv.load_dotenv()


class AskViridium:
    def __init__(self):
        self.constants = GlobalConstants()
        self.model_name = self.constants.model_name
        self.deployment_name = self.constants.deployment_name

        self.llm = AzureChatOpenAI(
            deployment_name=self.deployment_name,
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
        with open('modules/ask_viridium_ai/system_prompt_templates/newprompt.txt', 'r') as file:
            analysis_system_prompt = file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", analysis_system_prompt),
            ("human",
             "Material Name: {material}, manufactured by {manufacturer}. CONTEXT: used as {usecase}. Its chemical composition is: {chemical_composition}")
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

    def query(self, material_name, manufacturer_name: str = "Not Available", work_content: str = "Not Available"):
        material = material_name
        manufacturer = manufacturer_name
        work_content = work_content

        self.chemical_composition = self.cheminfo_chain.invoke(
            {"material": material, "example": self.constants.chemical_composition_example})
        chemicals_list = [chemical["name"] for chemical in self.chemical_composition["chemicals"]]

        self.result = self.analysis_chain.invoke(
            {"material": material, "manufacturer": manufacturer, "usecase": work_content,
             "chemical_composition": chemicals_list, "example": self.constants.analysis_example})

        return self.result

    def store(self):
        # saving results
        with open("compound_twostep.json", 'w') as file:
            json.dump(self.result, file)


if __name__ == '__main__':
    ask_vai = AskViridium()
    ans = ask_vai.query("Nitrogen, Cryogenic Liquid", "Matheson Tri-Gas, Inc.",
                        "Heat Treatment, Hipping, Annealing and Tempering")
