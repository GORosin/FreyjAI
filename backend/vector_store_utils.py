import os
import numpy as np
import re
import pandas as pd
from env import load_config, load_logging
from langchain_openai import ChatOpenAI
from thefuzz import fuzz
import time
import requests
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

config=load_config()
logger=load_logging(config)
small_model=ChatOpenAI(model="gpt-4o-mini",api_key=os.environ["OPENAI_API_KEY"])
token=os.environ["HUGGINGFACE_KEY"]
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"])

def embed_query(queries):
    """
    Use huggingface serverless api to get the embedding of a list of queries. The model is defined in the config file.
    Args:
        queries (list[str]): List of queries
    Returns:
        list[list[float]]: List of embeddings of each query
    Raises:
         ValueError: queries input is not a list
    """
    if not isinstance(queries,list):
        raise ValueError("input should be list of strings")
    model=config["PITCHBOOK_SETTINGS_LOCAL"]["embedding_model"]
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {f"Authorization": f"Bearer {token}","x-wait-for-model":"true"}
    payload={
	"inputs": queries,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
	

def find_best_match(match_str,match_list):
    """
    Find best item in match_list that matchs match_str, using fuzzy matching
    Args:
        match_str (str): string to match
        match_list (list): List of items to select
    Returns:
        (str,int): best matching item in list, match ratio
    """
    match_ratios=[fuzz.ratio(match_str,item) for item in match_list]
    return (match_list[np.argmax(match_ratios)],max(match_ratios))

def simple_query(query):
    """
    ask the llm for query if the query is not "N/A". This function is just used for parallilizing api calls
    Args:
        query (str): User query
    Returns:
        str: llm results of the query
    """
    llm_results="N/A" if query=="N/A" else small_model.invoke(query).content
    return llm_results


def embed_description(user_id,description,collection):
    """
    Create a chroma database of the company data. The documents are the business overviews. The metadata include company data, market cap, stock symbol
    Args:
        company_data (pandas.DataFrame): Dataframe of company data
    """
    chroma_client = chromadb.PersistentClient(path="profiles_db")
    collection = chroma_client.get_or_create_collection(name=collection,embedding_function=embed_query)


    metadata=[{"user_id":user_id}]
    collection.upsert(documents=[description],ids=ids,metadatas=metadata)


def match_user(user_id,n_results):
    """
    get peer companies, companies most like the company of the ticker. Queries chroma database, ticker should be in it
    Args:
         ticker (str): Company stock ticker to match other companies to
         n_results (int): Number of companies to return, defaults 5
    Returns:
         list: list of peer companies by ticker symbol
    """
    company=collection.get(include=["metadatas","documents"],where={"symbol":ticker})["documents"]
    peers=collection.query(query_texts=company,n_results=n_results,where={"symbol":{"$ne":ticker}})["metadatas"][0]
    peers=[meta["symbol"] for meta in peers]
    return peers

if __name__=="__main__":
    print(query_company("DXC"))
