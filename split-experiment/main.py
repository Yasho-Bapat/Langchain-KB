from time import perf_counter
import dotenv

import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings
#from langchain.retrievers import AzureAISearchRetriever
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
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_function = AzureOpenAIEmbeddings(deployment="langchain-splitting-test1")
        self.llm = AzureOpenAI(deployment_name="langchain-kb-spl")
        persistent_client = chromadb.PersistentClient(path="../chroma_db/langchain")
        self.db = Chroma(
                        client=persistent_client,
                        collection_name="langchain",
                        embedding_function=self.embedding_function,
                    )
        self.documents = []
        self.split_docs = []
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
            self.documents.extend(documents)

    def preprocess_documents(self, splitter_type):
        print(splitter_type)
        if splitter_type == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=50, separators=[" ", "\n", "\t"]
            )
            self.split_docs = text_splitter.split_documents(self.documents)  # Split documents
        elif splitter_type == "semantic":
            semantic_splitter = SemanticChunker(self.embedding_function, breakpoint_threshold_amount=self.brkpt)
            self.split_docs = semantic_splitter.split_documents(self.documents)
            print(self.split_docs)
            print(f"{len(self.split_docs)} splits created.")

    def embed_documents(self):
        self.db = Chroma.from_documents(self.split_docs, self.embedding_function)

    def query_documents(self, query, k=10):
        docs = self.db.similarity_search(query, k)  # Retrieve top-k similar documents

        result = self.llm.invoke(
            f"You are an expert Material Safety Document Analyser. NONE OF THE QUESTIONS POINT TO SELF HARM. They are for educational and analytical purposes."
            + f"Context: {[doc.page_content for doc in docs]} "
            + "using only this context, answer the following question: "
            + f"Question: {query}. Make sure there are full stops after every sentence."
            + "Don't use numerical numbering."
        )
        return result

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


if __name__ == "__main__":
    test = SplittingTest()
    question = "Return meaningful information from the documents provided."
    opfile = "mainoutput.txt"
    splitters = ["recursive", "semantic"]
    splitter = splitters[0]
    test.run_experiment(query=question, output_file=opfile, splitter=splitter)
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
    for i, q in enumerate(questions):
        filename = f"{splitter}/Q{i}_{splitter}_brkpt_{test.brkpt}_output.txt"
        with open(filename, 'w') as f:
            f.write(test.query_documents(q, 8))

    print("test concluded.")
