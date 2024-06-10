from time import perf_counter
import dotenv

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.retrievers import AzureAISearchRetriever
from langchain_openai import AzureOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()
import os


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
        pass

    def semantic_chunker(self):
        pass

    def section_aware_text_splitter(self):
        pass

    def no_splitters(self):
        pass


class SplittingTest:
    def __init__(self):
        self.docs_dir = "../docs"
        self.endpoint = os.getenv("ENDPOINT")
        self.embedding_function = AzureOpenAIEmbeddings()
        self.llm = AzureOpenAI(model="gpt-4o")
        self.documents = []

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

    def preprocess_documents(self, splitter_type):
        if splitter_type == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=50, separators=[" ", "\n", "\t"]
            )
            self.docs = text_splitter.split_documents(self.documents)  # Split documents
        elif splitter_type == "semantic":
            semantic_splitter = SemanticChunker(AzureOpenAIEmbeddings(deployment="langchain-kb-embedding-test1"))
            self.docs = semantic_splitter.split_documents(self.documents)
            print(self.docs)
            print(f"{len(self.docs)} splits created.")

    def embed_documents(self):
        self.db = Chroma.from_documents(self.docs, self.embedding_function)

    def query_documents(self, query, k=10):
        docs = self.db.similarity_search(query, k)  # Retrieve top-k similar documents

        result = self.llm.invoke(
            f"You are an expert Material Safety Document Analyser."
            + f"Context: {[doc.page_content for doc in docs]} "
            + "using only this context, answer the following question: "
            + f"Question: {query}. Make sure there are full stops after every sentence."
            + "Don't use numerical numbering."
        )
        return result.content

    def run_experiment(self, query, output_file, splitter):
        start = perf_counter()
        self.load_documents()
        self.preprocess_documents(splitter_type=splitter)
        self.embed_documents()
        preprocessed_time = perf_counter() - start
        result = self.query_documents(query)
        retrieval_time = perf_counter() - start - preprocessed_time
        print("ok")
        with open(output_file, "w") as file:
            file.write(result + "\n\n" + f"Langchain took: {retrieval_time} seconds")
        return result
