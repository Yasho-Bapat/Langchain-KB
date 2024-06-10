from time import perf_counter
import dotenv
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

questions = [
    "What are the Chemicals present in Vitrified Bonded STICK?",
    "What are the hazards associated with 341D Belts?",
    "What is the recommended action if the Ammonium Hydroxide is swallowed?",
    "What storage condition is recommended for this X?",
    "What first aid measure should be taken in case of skin contact with Havaklean KP?",
    "What type of eye protection is recommended when handling Copper Sulphate?",
    "What type of extinguishing media is suitable for a fire involving Citrisurf?",
    "What should be done to prevent from causing environmental contamination in the event of a spill?",
    "Is Vitrified Bonded WHEEL listed under the TSCA inventory?",
    "What is the UN number assigned to Argon Liquid for transport purposes?"
]


class Splitters:
    def __init__(self, text):
        self.text = text

    def recursive_character_text_splitter(self):
        splitter = RecursiveCharacterTextSplitter()
        return splitter.split_text(self.text)

    def semantic_chunker(self):
        splitter = SemanticChunker()
        return splitter.split_text(self.text)

    def section_aware_text_splitter(self):
        splitter = SemanticChunker()
        return splitter.split_text(self.text)

    def no_splitters(self):
        return [self.text]


class SplittingTest:
    def __init__(self, splitter_name):
        self.docs_dir = "../docs"
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_function = AzureOpenAIEmbeddings(deployment="langchain-splitting-test1")
        self.llm = AzureOpenAI(deployment_name="langchain-splitting-test")
        self.splitter_name = splitter_name

        self.db = Chroma(
            collection_name=f"langchain_{splitter_name}",
            embedding_function=self.embedding_function,
        )
        self.documents = []
        self.split_docs = []
        self.splitter = None

    def load_documents(self):
        file_paths = [
            os.path.join(self.docs_dir, file)
            for file in os.listdir(self.docs_dir)
            if file.endswith(".pdf")
        ]

        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            self.documents.extend(documents)

    def preprocess_documents(self):
        text_content = " ".join([doc.page_content for doc in self.documents])
        self.splitter = Splitters(text_content)

        if self.splitter_name == "recursive":
            self.split_docs = self.splitter.recursive_character_text_splitter()
        elif self.splitter_name == "semantic":
            self.split_docs = self.splitter.semantic_chunker()
        elif self.splitter_name == "section_aware":
            self.split_docs = self.splitter.section_aware_text_splitter()
        else:
            self.split_docs = self.splitter.no_splitters()

    def query_documents(self, query, k=10):
        docs = self.db.similarity_search(query, k)

        # code will change for Azure AI llm
        result = self.llm.invoke(
            f"You are an expert Material Safety Document Analyser."
            + f"Context: {[doc.page_content for doc in docs]} "
            + "using only this context, answer the following question: "
            + f"Question: {query}. Make sure there are full stops after every sentence."
            + "Don't use numerical numbering."
        )
        return result.content

    def run_experiment(self):
        start = perf_counter()
        self.load_documents()
        self.preprocess_documents()

        for question in questions:
            answer = self.query_documents(question, 8)
            filename = f"Q_{self.splitter_name}_output.txt"
            with open(filename, 'a') as f:
                f.write(f"Question: {question}\nAnswer: {answer}\n\n")

        end = perf_counter()
        print(f"Experiment with {self.splitter_name} completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    splitters = ["recursive", "semantic", "section_aware", "none"]

    for splitter in splitters:
        test = SplittingTest(splitter)
        test.run_experiment()

    print("All tests concluded.")
