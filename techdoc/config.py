import os
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse 


load_dotenv()

def get_llm(model: str = "us.anthropic.claude-sonnet-5", temperature: float = 0) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=model,
        region_name=os.environ["AWS_REGION"],
        temperature=temperature
    )

def get_judge_llm() -> ChatBedrockConverse:
    """LLM for LLM-as-judge evaluation: deterministic (temperature=0)."""
    return get_llm(model="us.anthropic.claude-sonnet-4-6", temperature=0)

def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
        region_name=os.environ["AWS_REGION"]
    )