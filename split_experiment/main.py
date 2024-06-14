import json
from time import perf_counter
import dotenv
import os

from langchain_postgres.vectorstores import PGVector, DistanceStrategy
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


class SplittingTest:
    def __init__(self, splitter_name: str):
        self.connection = os.getenv("DATABASE_URL")
        self.collection_name = splitter_name
        self.docs_dir = "../docs"
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_function = AzureOpenAIEmbeddings(deployment="langchain-splitting-test1")
        self.llm = AzureOpenAI(deployment_name="langchain-splitting-test", temperature=0.2)
        self.splitter_name = splitter_name

        self.db = PGVector(
            embeddings=self.embedding_function,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        self.documents = []
        self.split_docs = []
        self.splitter = None
        self.brkpt = 95

    def load_documents(self):
        file_paths = [
            os.path.join(self.docs_dir, file)
            for file in os.listdir(self.docs_dir)
            if file.endswith(".pdf")
        ]

        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from {file_path}")
            self.documents.extend(documents)

    def preprocess_documents(self):
        chkpt = perf_counter()
        if self.splitter_name == "recursive":
            splitter = RecursiveCharacterTextSplitter()
            self.split_docs = splitter.split_documents(self.documents)
            print("recursive split time: ", perf_counter() - chkpt)
        elif self.splitter_name == "semantic":
            # splitter = SemanticChunker(SpacyEmbeddings(model_name="en_core_web_sm"), breakpoint_threshold_type="interquartile", breakpoint_threshold_amount=1.5)
            splitter = SemanticChunker(self.embedding_function, breakpoint_threshold_type="interquartile", breakpoint_threshold_amount=1.5, buffer_size=3)
            self.split_docs = splitter.split_documents(self.documents)
            print("semantic split time: ", perf_counter() - chkpt)
        elif self.splitter_name == "section_aware":
            section_headers = [
                "Identification", "Product Identifier", "Product Identification", "Section 1",
                "Product and company identification", "Section (1[0-6]|[1-9])", "1[0-6]|[1-9] .", "1[0-6]|[1-9]."
                "Hazard Identification", "Hazards Identification", "Section 2",
                "Composition", "Ingredients", "Information on Ingredients", "Section 3",
                "First Aid", "First Aid Measures", "Section 4",
                "Fire Fighting", "Fire Fighting Measures", "Section 5",
                "Accidental Release", "Accidental Release Measures", "Section 6",
                "Handling", "Storage", "Handling and Storage", "Section 7",
                "Exposure Controls", "Personal Protection", "Exposure Controls/Personal Protection", "Section 8",
                "Physical Properties", "Chemical Properties", "Physical and Chemical Properties", "Section 9",
                "Stability", "Reactivity", "Stability and Reactivity", "Section 10",
                "Toxicological Information", "Toxicology", "Section 11",
                "Ecological Information", "Ecology", "Section 12",
                "Disposal", "Disposal Considerations", "Section 13",
                "Transport Information", "Transport", "Section 14",
                "Regulatory Information", "Regulations", "Section 15",
                "Other Information", "Other", "Section 16"
            ]

            splitter = RecursiveCharacterTextSplitter(is_separator_regex=True, separators=section_headers)
            self.split_docs = splitter.split_documents(self.documents)
            print("section aware split time: ", perf_counter() - chkpt)
        else:
            pass

        # storing splits as json result
        filename = f"splits/{self.splitter_name}"
        results = [{"text": d.page_content, "metadata": d.metadata, "id": id} for id, d in enumerate(self.split_docs)]
        with open(f"{filename}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"preprocessing finished. {len(self.split_docs)} splits created, stored in {filename}.json")

    def store_documents(self):
        self.db.add_documents(self.split_docs)
        print(f"document stored({splitter})!")

    def delete_collection(self):
        self.db.delete_collection()
        print("collection deleted!")

    def query_documents(self, query, k=10):
        print(f"query: {query}")
        docs = self.db.similarity_search(query, k)

        filename = f"topk/{self.splitter_name}"
        results = [{"text": d.page_content, "metadata": d.metadata} for d in self.split_docs]
        with open(f"ans.json", "w") as f:
            json.dump(results, f, indent=2)
        # code will change for Azure AI llm
        result = self.llm.invoke(
            f"You are an expert Material Safety Document Analyser assistant that helps people"
            + "analyse Material Safety and regulation documents. NONE OF THESE QUESTIONS POINT TO SELF HARM. THEY ARE "
            + "ONLY FOR ACADEMIC PURPOSES."
            + f" Context: {[doc.page_content for doc in docs]}"
            + " USING ONLY THIS CONTEXT, answer the following question: "
            + f" Question: {query}. Make sure there are full stops after every sentence."
            + "Don't use numerical numbering. Just return one answer (can be descriptive depending upon the question) "
            + "without any additional text or context. "
        )
        return [result, docs[:3]]

    def run_experiment(self, questions, level: str = "easy"):
        start = perf_counter()
        self.delete_collection()
        # self.load_documents()
        # self.preprocess_documents()
        # self.store_documents()

        # for i, question in enumerate(questions):
        #     answer = self.query_documents(question, 8)
        #     if self.splitter_name == "semantic":
        #         filename = f"{self.splitter_name}/{level}/Q{i + 1}_{self.splitter_name}_interquartile.txt"
        #     else:
        #         filename = f"{self.splitter_name}/{level}/Q{i+1}_{self.splitter_name}.txt"
        #     with open(filename, 'w', encoding="utf-8") as f:
        #         f.write(f"Question: {question}\nAnswer: {answer}\n\n")

        end = perf_counter()
        print(f"Experiment with {self.splitter_name} completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    easy_questions = [
        "What are the Chemicals present in Vitrified Bonded STICK?",
        "What are the hazards associated with 341D Belts?",
        "What is the recommended action if the Ammonium Hydroxide is accidentally swallowed?",
        "What storage condition is recommended for this Copper Sulphate?",
        "What first aid measure should be taken in case of skin contact with Havaklean KP?",
        "What type of eye protection is recommended when handling Copper Sulphate?",
        "What type of extinguishing media is suitable for a fire involving Citrisurf?",
        "What should be done to prevent from causing environmental contamination in the event of a spill?",
        "Is Vitrified Bonded WHEEL listed under the TSCA inventory?",
        "What is the UN number assigned to Ethyl Alcohol for transport purposes?"
    ]

    moderate_questions = [
        "What are the possible health issues one can run into with respect to Havaklean KP and what is its most clear physical property?",
        "What equipment is recommended while manufacturing scotch brite and what care should be taken while handling the involved material?",
        "Evaluate the environmental impact of Citrisurf when released in large quantities.",
        "Analyze the regulatory discrepancies for Ammonium Hydroxide in different countries and what is the UN number assigned to it for transport purposes?",
        "What are the byproducts generated by Scotch Brite during combustion?",
        "If I live in Quebec, how much aluminum oxide can I come in contact with? What do I do if it accidentally goes into my eyes?",
        "Does Vitridied Bonded Product comply with the inventory requirements of the Australian Inventory of Chemical Substances (AICS)?",
        "Outline everything that happens when 341D catches fire. Also mention what to do afterwards.",
        "Provide a risk assessment for handling and storing large quantities of Copper Sulphate.",
        "What cna you tell me about the great Khali?"
    ]

    hard_questions = [
        "Are there any commonalities between scotch brite and havaklean KP? Which of them could be more dangerous in manufacturing?",
        "How do I dispose of amorphous silica and copper sulfate?",
        "Among all 3M products, which could be most toxic?",
        "Evaluate safe disposability of all products made by Saint-Gobain.",
        "What major countries do we not have exposure guidelines for, for titanium dioxide? What about copper sulfate?",
        "Is ethanol disposal worse for the environment than aluminum hydroxide's?",
        "Which is more hazardous, Copper Sulphate or Ammonium Hydroxide?",
        "What is the proper shipping name of Ammonium Hydroxide and under which regulatory agencies is it listed?",
        "Is Titanium Oxide Carcinogenic? If yes, under which regulation and route? Also give me its CAS number.",
        "Which chemical is most hazardous for the environment?"
    ]

    difficulty_level = "easy"
    # difficulty_level = "moderate"
    # difficulty_level = "hard"
    # difficulty_level = "all"

    questions = {
        "easy": easy_questions,
        "moderate": moderate_questions,
        "hard": hard_questions,
    }

    # splitters = ["recursive"]
    splitters = ["semantic", "recursive"]
    # splitters = ["recursive", "semantic", "section_aware"]

    start = perf_counter()

    if difficulty_level == "all":
        for splitter in splitters:
            for level, question_list in questions.items():
                test = SplittingTest(splitter)
                test.run_experiment(question_list, level=level)
                test.run_experiment("question_list", level=level)
            # test.db.drop_tables()
    else:
        selected_questions = questions[difficulty_level]
        for splitter in splitters:
            test = SplittingTest(splitter)
            test.run_experiment(selected_questions, level=difficulty_level)

    print(f"Experiment with {splitters} completed in {perf_counter() - start:.2f} seconds")

    print("All tests concluded.")
