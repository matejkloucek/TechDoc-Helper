import os
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse 


load_dotenv()

def get_llm(model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0", temperature: float = 0) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=model,
        region_name=os.environ["AWS_REGION"],
        temperature=temperature
    )

def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
        region_name=os.environ["AWS_REGION"]
    )