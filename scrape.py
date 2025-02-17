import pandas as pd
import numpy as np
import tiktoken
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, broadcast
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st
import chromadb
from chromadb.config import Settings

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("api_key")

# Initialize the spark session
spark = SparkSession.builder \
.appName("Yelp Review") \
.getOrCreate()

# Loading the JSON files
business_df = spark.read.option("inferSchema", "true").json('data/yelp_academic_dataset_business.json')
review_df = spark.read.option("inferSchema", "true").json('data/yelp_academic_dataset_review.json')
user_df = spark.read.option("inferSchema", "true").json('data/yelp_academic_dataset_user.json')

# joining the business and review data
br_df = review_df.alias("r").join(broadcast(business_df.alias("b")), on = "business_id", how = "inner")

# calculating the average ratings per business
average_df = br_df.groupby(
    "b.business_id", "b.name", "b.categories", "b.address", "b.city", "b.state", "b.hours"
).agg(
    avg("r.stars").alias("avg_rating")
)

# convert the data to pandas dataframe for easy processing
restaurant_data = average_df.toPandas().dropna(subset = ["categories", "address"])
restaurant_data["description"] = (
    restaurant_data["name"] + "." +
    restaurant_data["categories"].astype(str) + "." +
    restaurant_data["address"].astype(str) + "," +
    restaurant_data["city"].astype(str) + "," +
    restaurant_data["state"].astype(str) 
)

# Initializing Chroma DB
chroma_client = chromadb.PersistentClient(
    path = 'chroma_cache',
    settings = chromadb.Settings(
        allow_reset = True,
        anonymized_telemetry = False
    )
)

# get or create collections
collections = chroma_client.get_or_create_collection(
    name = "restaurants",
    metadata = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 128,
        "hnsw:M": 16
    }
)

# Embedding generation through parallelization and caching
def processing():
    if not collections.count():
        embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

        #converting complex types to strings
        restaurant_data["hours"] = restaurant_data["hours"].apply(
            lambda x: str(x.asDict()) if x else ""
        )

        restaurant_data["categories"] = restaurant_data["categories"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x)
        )

        with ThreadPoolExecutor(max_workers = 10) as executor:
            # Batch processing
            batch_size = 100
            for i in range(0, len(restaurant_data), batch_size):
                batch = restaurant_data.iloc[i:i+batch_size]
                documents = batch["description"].tolist()
                ids = [str(x) for x in batch.index]

                #preparing chroma compatible metadata
                metadatas = []
                for _, row in batch.iterrows():
                    metadata = {
                        "name": str(row["name"]),
                        "avg_rating": float(row["avg_rating"]),
                        "categories": str(row["categories"]),
                        "city": str(row["city"]),
                        "state": str(row["state"]),
                        "hours": str(row["hours"])                        
                    }
                    metadatas.append(metadata)

                
                # Generate embeddings in parallel
                embedding_list = list(executor.map(embeddings.embed_documents, [documents]))

                # Add to chroma db
                collections.add(
                    embeddings = embedding_list[0],
                    documents = documents,
                    metadatas = metadatas,
                    ids = ids
                )
        chroma_client.persist()

processing() # Running it only once
llm = ChatOpenAI(
    model = "gpt-4-turbo-preview",
    temperature = 0.4,
    model_kwargs = {
        "response_format" : {"type": "json_object"},
        "seed": 42
    }
)
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

# Query optimization
def chroma_query(query, n_result = 50):
    query_embed = embeddings.embed_query(query)
    return collections.query(
        query_embeddings = [query_embed],
        n_results = n_result
    )

def recommend(query, cuisine_type = None, city = None, state = None, min_rating = 3.8, top_n = 5):
    # Phase 1: Vector Search
    result = chroma_query(query)

    # convert to pandas dataframe
    out = pd.DataFrame({
        "name":[meta.get("name", "") for meta in result["metadatas"][0]],
        "rating": [float(meta.get("avg_rating", 0)) for meta in result["metadatas"][0]],
        "categories": [meta.get("categories", "") for meta in result["metadatas"][0]],
        "city": [meta.get("city", "") for meta in result["metadatas"][0]],
        "state": [meta.get("state", "") for meta in result["metadatas"][0]],
        "hours": [eval(meta.get("hours", "{}")) if meta.get("hours") else {} for meta in result["metadatas"][0]],
        "distance": result["distances"][0]
    })

    # Phase 2: Filtering
    if cuisine_type:
        out = out[out["categories"].str.contains(cuisine_type, case = False)]
    if min_rating:
        out = out[out["rating"] >= min_rating]
    if city:
        out = out[out["city"].str.lower() == city.strip().lower()]
    if state:
        out = out[out["state"].str.upper() == state.strip().upper()] 
    
    # Phase 3: Final Ranking
    out = out.sort_values(by = ["distance", "rating"], ascending = [True, False]).head(top_n)

    # format hours for display
    def formatting(hours_dict):
        return "\n".join([f"{day}: {time}" for day, time in hours_dict.items()]) if hours_dict else "Not Available"
    
    formatted_data = []
    for _, row in out.iterrows():
        formatted_data.append({
            "name": row["name"],
            "rating": f"{row["rating"]:.1f}/5",
            "location":f"{restaurant_data.loc[int(row.name), 'city']}, {restaurant_data.loc[int(row.name), 'state']}",
            "hours": formatting(row["hours"])
        })
    # Generate responses
    response_shemas = [
        ResponseSchema(
            name = "restaurants",
            type = "list",
            description = "List of recommended restaurants with details"
        ),
        ResponseSchema(
            name = "disclaimer",
            type = "string",
            description = "Always detail 'Based on verified Yelp data'"
        )
    ]

    response_schema = [
        ResponseSchema(
            name = "name",
            type = "string",
            description = "Exact business name from Yelp database"
        ),
        ResponseSchema(
            name = "rating",
            type = "float",
            description = "Average rating rounded to one decimal place"
        ),
        ResponseSchema(
            name = "location",
            type = "string",
            description = "Combination of address, city and state"
        ),
        ResponseSchema(
            name = "hours",
            type = "String",
            description = "Formatted operating hours or 'Not available'"
        ),
        ResponseSchema(
            name = "yelp_id",
            type = "string",
            description = "Yelp business ID for verification"
        )
    ]
    main_parser = StructuredOutputParser.from_response_schemas(response_shemas)
    restaurant_parser = StructuredOutputParser.from_response_schemas(response_schema)
    prompt_template = """
    You are YelpGPT, an expert restaurant recommendation system. Follow these rules:

    1. Strict Data Adherence:
    - ONLY use data from the provided Yelp records
    - NEVER invent or assume details
    - For missing info: State "Not Available"

    2. Response Structure:
    {format_instructions}

    3. Content Requirements:
    - Include Yelp Business ID for verification
    - Highlight if results are filtered by location/rating
    - Add dietary alerts if mentioned in categories

    4. Safety Guidelines:
    - Disclose any content conflicts
    - Avoid subjective adjectives ("best", "amazing")
    - Maintain neutral, factual tone

    Input Data:
    {data}

    User Query:
    {query} 
    """
    format_instructions = restaurant_parser.get_format_instructions() + \
    "\n" + main_parser.get_format_instructions()
    yelp_prompt = PromptTemplate(
        template = prompt_template,
        input_variables = ["data", "query"],
        partial_variables = {"format_instructions": format_instructions}
    )
    chain = LLMChain(
        llm = llm, 
        prompt = yelp_prompt,
        output_parser = main_parser
    )
    return chain.invoke({
        "data": formatted_data,
        "query": "Restaurants in Massachusetts."
    })

# Streamlit response
def main():
    st.title("🍔 YELP AI restaurant finder")

    with st.form("search_form"):
        query = st.text_input("Please state your desired spot")
        cuisine = st.text_input("Any preferences for cuisine? (Optional)")
        city_filter = st.text_input("City (optional): ")
        state_filter = st.text_input("State abbreviation (Optional): ", max_chars = 2)
        min_rating = st.slider("Minimum rating:", 3.0, 5.0, 3.8)
        submitted = st.form_submit_button("Find Restaurants!")
    
    if submitted:
        with st.spinner("Analyzing reviews..."):
            results = recommend(query = query, cuisine_type = cuisine, min_rating = min_rating, city = city_filter, state = state_filter.upper() if state_filter else None)
            st.success("🎉 Top Recommendations:")
            st.markdown(results)
if __name__ == "__main__":
    main()
    spark.stop()
