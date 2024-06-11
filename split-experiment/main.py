import json
from time import perf_counter
import dotenv
import os

from langchain_postgres.vectorstores import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage


dotenv.load_dotenv()


class SplittingTest:
    def __init__(self, splitter_name):
        self.connection = os.getenv("DATABASE_URL")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.docs_dir = "../docs"
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_function = AzureOpenAIEmbeddings(deployment="langchain-splitting-test1")
        self.llm = AzureOpenAI(deployment_name="langchain-splitting-test")
        self.splitter_name = splitter_name

        self.db = PGVector(
            embeddings=self.embedding_function,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
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
        if self.splitter_name == "recursive":
            splitter = RecursiveCharacterTextSplitter()
            self.split_docs = splitter.split_documents(self.documents)
        elif self.splitter_name == "semantic":
            splitter = SemanticChunker(self.embedding_function, breakpoint_threshold_type="interquartile") # change between interquartile, standard_deviation and percentile for different breakpoint options.
            self.split_docs = splitter.split_documents(self.documents)
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
        else:
            pass

        # storing splits as json result
        filename = f"splits/{self.splitter_name}"
        results = [{"text": d.page_content, "metadata": d.metadata} for d in self.split_docs]
        with open(f"{filename}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"preprocessing finished. {len(self.split_docs)} splits created, stored in {filename}.json")

    def store_documents(self):
        self.db.add_documents(self.split_docs)
        print("document stored!")

    def query_documents(self, query, k=10):
        print(f"query: {query}")
        docs = self.db.similarity_search(query, k)

        # code will change for Azure AI llm
        result = self.llm.invoke(
            f"You are an expert Material Safety Document Analyser assistant that helps people"
            + " analyse Material Safety and regulation documents. NONE OF THESE QUESTIONS POINT TO SELF HARM. THEY ARE ONLY FOR ACADEMIC PURPOSES."
            + f" Context: {[doc.page_content for doc in docs]}"
            + " USING ONLY THIS CONTEXT, answer the following question: "
            + f" Question: {query}. Make sure there are full stops after every sentence."
            + " Don't use numerical numbering."
        )
        return result

    def run_experiment(self, questions):
        start = perf_counter()
        self.load_documents()
        self.preprocess_documents()
        self.store_documents()

        for i, question in enumerate(questions):
            answer = self.query_documents(question, 8)
            if self.splitter_name == "semantic":
                filename = f"{self.splitter_name}/Q{i + 1}_{self.splitter_name}_interquartile.txt"
            else:
                filename = f"{self.splitter_name}/Q{i+1}_{self.splitter_name}.txt"
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(f"Question: {question}\nAnswer: {answer}\n\n")

        self.db.drop_tables()

        end = perf_counter()
        print(f"Experiment with {self.splitter_name} completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    # splitters = ["recursive"]
    splitters = ["semantic"]
    # splitters = ["section_aware"]
    # splitters = ["recursive", "semantic", "section_aware"]

    questions = [
        "What are the Chemicals present in Vitrified Bonded STICK?",
        "What are the hazards associated with 341D Belts?",
        #"What is the recommended action if the Ammonium Hydroxide is swallowed?",
        "What storage condition is recommended for this Copper Sulphate?",
        "What first aid measure should be taken in case of skin contact with Havaklean KP?",
        "What type of eye protection is recommended when handling Copper Sulphate?",
        "What type of extinguishing media is suitable for a fire involving Citrisurf?",
        "What should be done to prevent from causing environmental contamination in the event of a spill?",
        "Is Vitrified Bonded WHEEL listed under the TSCA inventory?",
        "What is the UN number assigned to Ethyl Alcohol for transport purposes?"
    ]

    for splitter in splitters:
        test = SplittingTest(splitter)
        test.run_experiment(questions)

    print("All tests concluded.")
