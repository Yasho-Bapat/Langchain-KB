import dotenv
from langchain_openai import AzureOpenAI

dotenv.load_dotenv()

llm = AzureOpenAI(
    deployment_name="langchain-kb-spl",
)

result = llm.invoke("Tell me a joke")

print(result)
