import dotenv
import json
from typing import Optional, List
import time

from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.callbacks import get_openai_callback

from askviridium.app.global_constants import GlobalConstants
from askviridium.app.models import ChemicalComposition, MaterialInfo
from .tracking import Logger

dotenv.load_dotenv()


class AskViridium:
    def __init__(self):
        self.logger = Logger()
        self.loginfo = dict()
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
        self.pfas = None

        self.result = str()

    def prompt1_init(self):
        with open('ask_viridium_ai/findchemicals_prompt.txt', 'r') as file:
            cheminfo_system_prompt = file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", cheminfo_system_prompt),
            ("human", "Material Name: {material}"),
        ])
        return prompt

    def prompt2_init(self):
        with open('ask_viridium_ai/newprompt.txt', 'r') as file:
            analysis_system_prompt = file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", analysis_system_prompt),
            ("human",
             "Material Name: {material}, manufactured by {manufacturer}. CONTEXT: used as {usecase}. Its chemical composition is: {chemical_composition}. Additional info: {additional_info}")
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

    def log(self, time, material_name, manufacturer_name, tokens_for_cheminfo, tokens_for_analysis, cost_cheminfo, cost_analysis, chemlist):
        self.loginfo["time"] = time
        self.loginfo["material_name"] = material_name
        self.loginfo["manufacturer_name"] = manufacturer_name
        self.loginfo["tokens_used_for_chemical_composition"] = tokens_for_cheminfo
        self.loginfo["cost_chemical_composition"] = cost_cheminfo
        self.loginfo["tokens_used_for_analysis"] = tokens_for_analysis
        self.loginfo["cost_analysis"] = cost_analysis
        self.loginfo["total_cost"] = cost_analysis + cost_cheminfo
        self.loginfo["chemical_composition"] = str(chemlist)
        self.loginfo["PFAS_status"] = self.pfas
        self.loginfo["user_id"] = "umesh" # placeholder

        self.logger.log(info=self.loginfo)

    def query(self, material_name, manufacturer_name: Optional[str] = "Not Available", work_content: Optional[str] = "Not Available", additional_info: Optional[str] = None):
        material = material_name
        manufacturer = manufacturer_name
        work_content = work_content
        rn = time.time()

        with get_openai_callback() as cb:
            self.chemical_composition = self.cheminfo_chain.invoke(
                {"material": material, "example": self.constants.chemical_composition_example})
            chemicals_list = [chemical["name"] for chemical in self.chemical_composition["chemicals"]]
            tokens_for_cheminfo = cb.total_tokens
            cost_for_cheminfo = cb.total_cost

        with get_openai_callback() as cb:
            self.result = self.analysis_chain.invoke(
                {"material": material, "manufacturer": manufacturer, "usecase": work_content,
                 "chemical_composition": chemicals_list, "example": self.constants.analysis_example, 
                 "additional_info": None})
            tokens_for_analysis = cb.total_tokens
            cost_for_analysis = cb.total_cost
            self.pfas = self.result["decision"]

        self.log(rn, material_name, manufacturer_name, tokens_for_cheminfo, tokens_for_analysis, cost_for_cheminfo, cost_for_analysis, chemicals_list)
        self.logger.save()

        store = self.store()
        print(store)

        return self.result

    def handle_user_query(self, additional_info, material, manufacturer, work_content, chemicals_list):
        with get_openai_callback() as cb:
            self.result = self.analysis_chain.invoke(
                {"material": material, "manufacturer": manufacturer, "usecase": work_content,
                 "chemical_composition": chemicals_list, "example": self.constants.analysis_example,
                 "additional_info": additional_info})
    def store(self):
        # Read existing data from the file
        try:
            with open("data.json", 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []

        # Append the new result
        data.append(self.result)

        # Write the updated data back to the file
        with open("data.json", 'w') as file:
            json.dump(data, file, indent=4)

        return "results saved"


if __name__ == '__main__':
    ask_vai = AskViridium()
    ans = ask_vai.query("Nitrogen, Cryogenic Liquid", "Matheson Tri-Gas, Inc.",
                        "Heat Treatment, Hipping, Annealing and Tempering")
