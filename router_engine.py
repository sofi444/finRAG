from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine

#from llama_index.llms.openai import OpenAI
#from llama_index.embeddings.openai import OpenAIEmbedding

import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



''' Load data and parse with LlamaParse '''
# parse pdf files into markdown
# up to 1000 pages per day for free with LlamaParse
parser = LlamaParse(result_type="markdown", verbose=True)
documents = SimpleDirectoryReader(
    #"./data",
    "./data/small_sample",
    file_extractor={".pdf": parser}
).load_data(show_progress=True)


''' LLM setup'''
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=5000,
    max_new_tokens=512,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.1},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)


#llm = OpenAI(model="gpt-3.5-turbo-0125")
#embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


''' Index setup '''
vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)

#engine_compact = vector_index.as_query_engine(response_mode="compact")
#engine_refine = vector_index.as_query_engine(response_mode="refine")
#engine_treesumm = vector_index.as_query_engine(response_mode="tree_summarize")

vector_tool = QueryEngineTool(
    vector_index.as_query_engine(response_mode="refine"),
    metadata=ToolMetadata(
        name="vector_search",
        description="Best suited for retrieving specific information."
    )
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Best suited for summarising larger context (e.g. entire doc)."
    )
)

engine_router = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    select_multi=True,
)

query = "What is the aggregate market value of the registrant's common stock held by non-affiliates as of December 31, 2021?"
query2 = "Compared to 2021, how have Microsoft's challenges changed?"

response = engine_router.query(query)
response2 = engine_router.query(query2)

print(f"Query1: {query}\n\nResponse1: {response}\n\n")
print(f"Query2: {query2}\n\nResponse2: {response2}\n\n")