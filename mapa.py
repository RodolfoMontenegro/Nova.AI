'''
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.
...
END OF TERMS AND CONDITIONS

Copyright 2024 Rodolfo Montenegro Ochoa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import psycopg2
import ollama
import chromadb
import logging
import json
import re
import os
import sqlvalidator
import hashlib
import requests
import langdetect
import difflib
import spacy
import pandas as pd
from httpx import Timeout, ConnectTimeout, HTTPStatusError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, session
from logging.config import dictConfig
from langdetect import detect
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from tenacity import retry, stop_after_attempt, wait_fixed
from contextlib import contextmanager
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from spacy.matcher import PhraseMatcher

# Setup logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)

secret_key = os.urandom(24)
app.secret_key = secret_key

ALLOWED_EXTENSIONS = {'pdf', 'txt'}
UPLOAD_FOLDER = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load English and Spanish spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

# General concept keywords in both English and Spanish
general_concepts = {
    "en": ["what", "explain", "define"],
    "es": ["qué", "explicar", "definir"]
}

# Greeting words in both languages
greetings_en = [
    "hello", "hi", "hey", "greetings", "hello!", "hi there", "hey there", "howdy", "what's up",
    "good morning", "good afternoon", "good evening", "good night"
]
greetings_es = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "saludos",
    "ola, ¿qué tal?", "¿cómo estás?", "¿qué pasa?", "¿qué hay?", "sup", "buenas", "muy buenas", "¿qué onda?", "¿qué tal?"
]

# Initialize PhraseMatcher for English and Spanish
matcher_en = PhraseMatcher(nlp_en.vocab)
matcher_es = PhraseMatcher(nlp_es.vocab)

# Add greeting patterns in English and Spanish
patterns_en = [nlp_en(text) for text in greetings_en]
patterns_es = [nlp_es(text) for text in greetings_es]
matcher_en.add("GREETING_EN", patterns_en)
matcher_es.add("GREETING_ES", patterns_es)

def is_greeting(text, language):
    """Detect if the text is a greeting based on the language."""
    logging.info(f"Checking if '{text}' is a greeting in {language}...")

    if language == "es":
        doc = nlp_es(text)
        matches = matcher_es(doc)
        logging.info(f"Greeting matches in Spanish: {matches}")
        if len(matches) > 0:
            return True
        # Fallback to simple string matching
        return text.lower() in greetings_es
    else:
        doc = nlp_en(text)
        matches = matcher_en(doc)
        logging.info(f"Greeting matches in English: {matches}")
        if len(matches) > 0:
            return True
        return text.lower() in greetings_en

def is_general_concept(text, language):
    """Detect if the text is a general concept question (e.g., asking for explanations)."""
    logging.info(f"Checking if '{text}' is a general concept in {language}...")

    if language == "es":
        doc = nlp_es(text)
    else:
        doc = nlp_en(text)

    # Check for the presence of general keywords in the sentence
    general_concepts_list = general_concepts.get(language, [])
    for token in doc:
        if token.text.lower() in general_concepts_list:
            logging.info(f"Found general concept keyword: {token.text}")
            return True
    return False

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5438'
}

# Global variables for client, ef, and collection
client = None
ef = None
collection = None
ddl_collection = None
sql_collection = None
documentation_collection = None

# Initialize the connection pool globally
db_pool = SimpleConnectionPool(minconn=1, maxconn=10, **DB_PARAMS)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _get_db_connection():
    """Get a connection from the connection pool."""
    try:
        conn = db_pool.getconn()
        if conn:
            logging.info("Database connection retrieved from pool.")
            return conn
        else:
            logging.error("Failed to retrieve a database connection from pool.")
            raise psycopg2.OperationalError("No connection available in pool.")
    except Exception as e:
        logging.error(f"Error retrieving connection from pool: {e}")
        raise

@contextmanager
def get_db_connection():
    """Context manager for getting a connection from the pool."""
    conn = None
    try:
        conn = db_pool.getconn()
        logging.info("Database connection retrieved from pool.")
        yield conn
    finally:
        if conn:
            if not conn.closed:
                db_pool.putconn(conn)
                logging.info("Database connection returned to pool.")
            else:
                logging.warning("Connection was closed; not returning to pool.")

def init_chromadb():
    global client, ef, ddl_collection, sql_collection, documentation_collection  # Declare all globals

    try:
        # Create a ChromaDB persistent client
        client = chromadb.PersistentClient(path="./chromadb")

        # Create an OllamaEmbeddingFunction with a custom endpoint
        ef = OllamaEmbeddingFunction(
            model_name="bge-m3",
            url="http://localhost:11434/api/embed",
        )

        # Initialize all collections
        ddl_collection = client.get_or_create_collection(name="ddl_collection", embedding_function=ef)
        sql_collection = client.get_or_create_collection(name="sql", embedding_function=ef)
        documentation_collection = client.get_or_create_collection(name="documentation", embedding_function=ef)

        logging.info("ChromaDB client and collections initialized successfully.")

    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB: {e}")
        raise SystemExit("Failed to initialize ChromaDB. Exiting application.")

# Initialize ChromaDB at application startup
init_chromadb()

def extract_sql(llm_response):
    if not isinstance(llm_response, str):
        raise ValueError("The LLM response should be a string")

    llm_response = llm_response.replace("\\_", "_").replace("\\", "")
    sql = re.search(r"```sql\n((.|\n)*?)(?=;|\[|```)", llm_response, re.DOTALL)
    select_with = re.search(r'(select|(with.*?as \())(.*?)(?=;|\[|```)', llm_response, re.IGNORECASE | re.DOTALL)

    if sql:
        logging.info(f"Extracted SQL: {sql.group(1)}")
        return sql.group(1).replace("```", "")
    elif select_with:
        logging.info(f"Extracted SQL: {select_with.group(0)}")
        return select_with.group(0)
    else:
        return llm_response

def generate_ddl_id(ddl):
    """Generate a unique ID for a DDL based on its content."""
    return hashlib.md5(ddl.encode('utf-8')).hexdigest()

def embed_ddl(ddls):
    """Embed the DDLs using Ollama embeddings and store them in ChromaDB."""
    global client, ef, ddl_collection  # Use ddl_collection instead of collection

    try:
        if not client or not ddl_collection:
            logging.error("ChromaDB client or ddl_collection is not initialized.")
            raise RuntimeError("ChromaDB client or ddl_collection is not initialized.")

        for ddl in ddls:
            ddl_id = generate_ddl_id(ddl)
            existing = ddl_collection.get(ids=[ddl_id])
            if existing['ids']:
                logging.warning(f"Embedding with ID {ddl_id} already exists. Skipping...")
                continue

            response = ollama.embeddings(model="bge-m3", prompt=ddl)
            embedding = response["embedding"]
            ddl_collection.add(
                ids=[ddl_id],
                embeddings=[embedding],
                documents=[ddl]
            )
        logging.info("DDLs embedded and stored in ChromaDB.")
    except ConnectTimeout:
        logging.error("Connection timeout while embedding DDLs.")
    except HTTPStatusError as e:
        logging.error(f"HTTP error while embedding DDLs: {e}")
    except Exception as e:
        logging.error(f"Error embedding DDLs: {e}")

def chat_with_bot(question):
    try:
        # Step 1: Configure and initialize the Ollama client
        ollama_config = {
            "model": "llama3.1",
            "ollama_host": "http://localhost:11434"
        }
        ollama_client, model, ollama_options, keep_alive = init_ollama(ollama_config)

        # Step 2: Analyze the question and decide which database to query
        decision_prompt = f"Analyze the question and decide if it should query the PostgreSQL database or ChromaDB: {question}"
        decision_response = submit_prompt(ollama_client, model, decision_prompt, ollama_options, keep_alive)

        # Ensure response is a dictionary
        if isinstance(decision_response, str):
            decision_response = json.loads(decision_response)

        decision_content = decision_response.get("message", {}).get("content", "").lower()

        if "chromadb" in decision_content:
            logging.info("Bot decided to query ChromaDB.")

            # Query ChromaDB (ddl_collection as default)
            chromadb_response = query_chromadb(question, collection_name="ddl_collection")
            if chromadb_response:
                logging.info(f"Retrieved DDL structure from ChromaDB: {chromadb_response}")

                # Check if the DDL structure contains a 'geom' column for spatial data
                if 'geom' in chromadb_response.lower():
                    logging.info("Detected 'geom' column in DDL. Routing to spatial query logic.")

                    # Retrieve relevant spatial documentation to generate a PostGIS SQL query
                    spatial_docs_response = query_chromadb(question, collection_name="documentation_collection")
                    if spatial_docs_response:
                        logging.info(f"Retrieved relevant spatial documentation: {spatial_docs_response}")

                        # Generate SQL query using the retrieved documentation
                        sql_translation_prompt = f"Using the following PostGIS documentation: {spatial_docs_response}, translate this natural language query to a PostGIS SQL query: {question}"
                        sql_translation_response = submit_prompt(ollama_client, model, sql_translation_prompt, ollama_options, keep_alive)

                        if isinstance(sql_translation_response, str):
                            sql_translation_response = json.loads(sql_translation_response)

                        sql_query = extract_sql(sql_translation_response.get("message", {}).get("content", ""))

                        if not sql_query:
                            logging.error("Failed to extract a valid SQL query.")
                            return "No valid SQL query was generated."

                        logging.info(f"Extracted SQL query: {sql_query}")

                        # Step 3: Execute the SQL query using the connection pool
                        with get_db_connection() as conn:
                            results = query_db(conn, sql_query)
                            if results:
                                # Ingest successful SQL query into the SQL collection
                                ingest_sql_into_chromadb(sql_query, question, collection_name="sql_collection")
                                return f"The query returned the following results: {results}"
                            else:
                                return "No relevant data found in the PostgreSQL database."
                    else:
                        return "No relevant spatial documentation found to generate a query."

                # If no 'geom' column is found, continue with standard SQL generation
                logging.info("No 'geom' column detected. Proceeding with standard SQL generation.")
                sql_translation_prompt = f"Using the following DDL structure: {chromadb_response}, translate this natural language query to SQL: {question}"
                sql_translation_response = submit_prompt(ollama_client, model, sql_translation_prompt, ollama_options, keep_alive)

                if isinstance(sql_translation_response, str):
                    sql_translation_response = json.loads(sql_translation_response)

                sql_query = extract_sql(sql_translation_response.get("message", {}).get("content", ""))

                if not sql_query:
                    logging.error("Failed to extract a valid SQL query.")
                    return "No valid SQL query was generated."

                logging.info(f"Extracted SQL query: {sql_query}")

                # Step 3: Execute the SQL query using the connection pool
                with get_db_connection() as conn:
                    results = query_db(conn, sql_query)
                    if results:
                        # Ingest successful SQL query into the SQL collection
                        ingest_sql_into_chromadb(sql_query, question, collection_name="sql_collection")
                        return f"The query returned the following results: {results}"
                    else:
                        return "No relevant data found in the PostgreSQL database."
            else:
                return "No relevant information found in ChromaDB."

        elif "postgresql" in decision_content or "sql" in decision_content:
            logging.info("Bot decided to query the PostgreSQL database.")

            # Generate and execute SQL directly
            sql_translation_prompt = f"Translate this natural language query to SQL: {question}"
            sql_translation_response = submit_prompt(ollama_client, model, sql_translation_prompt, ollama_options, keep_alive)

            if isinstance(sql_translation_response, str):
                sql_translation_response = json.loads(sql_translation_response)

            sql_query = extract_sql(sql_translation_response.get("message", {}).get("content", ""))

            if not sql_query:
                logging.error("Failed to extract a valid SQL query.")
                return "No valid SQL query was generated."

            logging.info(f"Extracted SQL query: {sql_query}")

            # Execute the SQL query using the connection pool
            with get_db_connection() as conn:
                results = query_db(conn, sql_query)
                if results:
                    # Ingest successful SQL query into the SQL collection
                    ingest_sql_into_chromadb(sql_query, question, collection_name="sql_collection")
                    return f"The query returned the following results: {results}"
                else:
                    return "No relevant data found in the PostgreSQL database."

        else:
            logging.warning("The bot's decision was unclear.")
            return "I couldn't understand where to search for the information. Please try rephrasing your question."

    except Exception as e:
        logging.error(f"Error chatting with the bot: {e}")
        return "An unexpected error occurred while processing your query."

def chat_with_bot_and_learn(question, chat_history=[]):
    """Interact with the bot using a RAG model, maintaining chat history for learning."""
    try:
        # Manually detect common greetings first
        if question.lower() in greetings_es:
            logging.info(f"'{question}' is recognized as a Spanish greeting.")
            return "¡Hola! Parece que me estás saludando. ¿Cómo puedo asistirte con datos o análisis hoy?"
        elif question.lower() in greetings_en:
            logging.info(f"'{question}' is recognized as an English greeting.")
            return "Hello! It seems like you're greeting me. How can I assist you with data or analysis today?"

        # If it's not a common greeting, detect the language using langdetect
        try:
            language = langdetect.detect(question)
        except langdetect.lang_detect_exception.LangDetectException:
            language = "en"  # Default to English if detection fails

        logging.info(f"Detected language: {language}")

        # Check if it's a greeting using PhraseMatcher
        if is_greeting(question, language):
            logging.info(f"'{question}' is a greeting in {language}.")
            if language == "es":
                return "¡Hola! Parece que me estás saludando. ¿Cómo puedo asistirte con datos o análisis hoy?"
            else:
                return "Hello! It seems like you're greeting me. How can I assist you with data or analysis today?"

        # Step 1: Initialize the Ollama client
        ollama_config = {
            "model": "llama3.1",
            "ollama_host": "http://localhost:11434"
        }
        ollama_client, model, ollama_options, keep_alive = init_ollama(ollama_config)

        # Step 2: Check if it's a general concept question
        if is_general_concept(question, language):
            # Generate a direct response using LLM
            analysis_prompt = f"Provide a detailed response to the following question: {question}"
            analysis_response = submit_prompt(ollama_client, model, analysis_prompt, ollama_options, keep_alive)

            if isinstance(analysis_response, str):
                analysis_response = json.loads(analysis_response)

            response = analysis_response.get("message", {}).get("content", "Could not generate a response.")

            # No need to translate as the model supports multiple languages
            # Just return the response
            return response

        # Step 3: If it's SQL-related or data-driven, query ChromaDB
        all_collections = ["ddl_collection", "sql_collection", "documentation_collection"]
        retrieved_information = []

        for collection_name in all_collections:
            collection_response = query_chromadb_collection(question, collection_name)
            
            # Ensure the response is valid before trying to access it
            if collection_response:
                retrieved_information.append(f"From {collection_name} collection:\n{collection_response}")

        # Step 4: Generate a detailed response using the retrieved information if available
        if retrieved_information:
            combined_information = "\n\n".join(retrieved_information)
            analysis_prompt = f"Based on the following information, provide a detailed answer to the question: {question}\n\n{combined_information}"
        else:
            # If no relevant info found in ChromaDB, generate the answer from the LLM
            analysis_prompt = f"Provide a detailed response to the following question: {question}"

        analysis_response = submit_prompt(ollama_client, model, analysis_prompt, ollama_options, keep_alive)
        
        if isinstance(analysis_response, str):
            analysis_response = json.loads(analysis_response)

        response = analysis_response.get("message", {}).get("content", "Could not generate a response.")
        
        # Return the response as-is, no translation necessary
        return response

    except Exception as e:
        logging.error(f"Error in chat_with_bot_and_learn: {e}")
        return "An error occurred while processing your query."

def embed_chat_history(chat_history, ollama_client, model):
    """Embed the chat history into the `documentation_collection` for learning purposes."""
    global client, documentation_collection
    
    try:
        if not client or not documentation_collection:
            logging.error("ChromaDB client or documentation collection is not initialized.")
            return

        # Convert chat history to a string format suitable for embedding
        history_text = "\n".join([f"User: {entry['user_input']}\nBot: {entry['bot_response']}" for entry in chat_history])

        # Generate a document ID based on the chat history
        doc_id = hashlib.md5(history_text.encode('utf-8')).hexdigest()

        # Check if this chat history already exists
        existing = documentation_collection.get(ids=[doc_id])
        if existing['ids']:
            logging.info(f"Chat history with ID {doc_id} already exists. Skipping embedding.")
            return

        # Generate embedding using the correct model (ensure correct dimensionality)
        response = ollama_client.embeddings(
            model="bge-m3",  # Ensure this matches your embedding model
            prompt=history_text
        )

        embedding = response.get("embedding")
        
        # Ensure embedding matches the collection's expected dimensionality
        if len(embedding) != 1024:  # Adjust this check based on your collection's dimensionality
            logging.error(f"Embedding dimension {len(embedding)} does not match collection dimensionality 1024.")
            return

        if embedding:
            # Store the text and its embedding in `documentation_collection`
            documentation_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[history_text]
            )
            logging.info(f"Chat history successfully embedded into `documentation_collection`.")
        else:
            logging.error("Failed to generate embedding for chat history.")

    except Exception as e:
        logging.error(f"Error embedding chat history: {e}")

def query_chromadb_collection(prompt, collection_name):
    """Query a specific ChromaDB collection using the provided prompt."""
    try:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)
        if not collection:
            logging.error(f"Collection {collection_name} not found in ChromaDB.")
            return None

        # Generate embedding for the prompt
        response = ollama.embeddings(
            prompt=prompt,
            model="bge-m3"
        )

        # Perform the query on the specified ChromaDB collection
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1  # Fetch the most relevant document
        )

        if results['ids']:
            # Check if documents array is non-empty and well-formed
            if len(results['documents']) > 0 and len(results['documents'][0]) > 0:
                data = results['documents'][0][0]
                logging.info(f"Retrieved relevant information from {collection_name} collection: {data}")
                return data
            else:
                logging.warning(f"Documents array is empty in {collection_name} collection.")
                return None
        else:
            logging.warning(f"No matching document found in {collection_name}.")
            return None

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response from {collection_name} collection: {e}")
        return None
    except Exception as e:
        logging.error(f"Error querying {collection_name} collection: {e}")
        return None


def apply_schema_to_sql(conn, sql_query):
    """Apply detected schemas to the table names in the SQL query."""
    try:
        # Clean up the SQL query by removing leading/trailing whitespace and replacing newlines with spaces
        sql_query = sql_query.replace('\n', ' ').strip()

        # Extract table names from the cleaned SQL query
        table_pattern = re.compile(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
        tables = table_pattern.findall(sql_query)

        logging.info(f"Tables found in SQL query: {tables}")

        # Detect and apply schema for each table
        for table in tables:
            schema = detect_schema_for_table(conn, table)
            if schema:
                # Replace the table name with schema.table_name in the SQL query
                sql_query = re.sub(rf'\b{table}\b', f'{schema}.{table}', sql_query)
                logging.info(f"Applying schema '{schema}' to table '{table}'")

        # Final cleanup of the SQL query
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()

        logging.info(f"SQL query after schema application: {sql_query}")
        return sql_query

    except Exception as e:
        logging.error(f"Error applying schema to SQL query: {e}")
        return sql_query  # Return the original query if something goes wrong

def query_db(conn, query, params=None):
    """Apply schema to SQL query and execute it on the database."""
    try:
        query = apply_schema_to_sql(conn, query)
        query = query.replace("\n", " ").strip()
        logging.debug(f"Final cleaned SQL query: {query}")

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            logging.info(f"Executing query: {query}")
            cursor.execute(query, params)
            results = cursor.fetchall()
            logging.info(f"Query executed successfully: {query}")
            return results
    except psycopg2.Error as e:
        logging.error(f"Database error executing query: {e.pgcode} - {e.pgerror}")
        return None
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        return None

def query_chromadb(prompt, collection_name="ddl_collection"):
    """Query the specified ChromaDB collection using the prompt."""
    global client, ef  # Ensure global client and embedding function are accessible

    try:
        if not client:
            raise RuntimeError("ChromaDB client is not initialized.")
        
        # Fetch the specified collection by name
        collection = client.get_collection(name=collection_name, embedding_function=ef)
        if not collection:
            raise RuntimeError(f"ChromaDB collection '{collection_name}' is not initialized.")

        # Generate embedding for the prompt
        response = ollama.embeddings(
            prompt=prompt,
            model="bge-m3"
        )

        # Perform the query on the specified ChromaDB collection
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1  # Fetch the most relevant document
        )

        if results['ids']:
            data = results['documents'][0][0]
            logging.info(f"Retrieved data from {collection_name}: {data}")
            return data
        else:
            logging.warning(f"No matching document found in {collection_name}.")
            return None

    except Exception as e:
        logging.error(f"Error querying {collection_name} collection: {e}")
        return f"Error occurred while querying {collection_name}: {e}"

# Initialize the Ollama client
def init_ollama(config):
    try:
        ollama = __import__("ollama")
    except ImportError:
        raise DependencyError(
            "You need to install required dependencies to execute this method, run command:"
            " \npip install ollama"
        )

    host = config.get("ollama_host", "http://localhost:11434")
    model = config["model"]
    if ":" not in model:
        model += ":latest"

    ollama_client = ollama.Client(host, timeout=Timeout(240.0))
    keep_alive = config.get('keep_alive', None)
    ollama_options = config.get('options', {})

    # Ensure model is available
    model_list_response = ollama_client.list()
    model_list = [model_element['model'] for model_element in model_list_response.get('models', [])]
    if model not in model_list:
        ollama_client.pull(model)

    return ollama_client, model, ollama_options, keep_alive

# Function to submit prompt to Ollama
def submit_prompt(ollama_client, model, prompt, ollama_options, keep_alive=None):
    logging.info(f"Ollama parameters: model={model}, options={ollama_options}, keep_alive={keep_alive}")
    logging.info(f"Prompt Content: {json.dumps(prompt)}")

    try:
        # Call the Ollama API
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options=ollama_options,
            keep_alive=keep_alive
        )

        # Check the type of response
        if isinstance(response, dict):
            # If response is a dict, look for 'content' or other relevant keys
            if 'content' in response:
                final_response = response['content']
            else:
                final_response = json.dumps(response)  # Serialize dict to a string
            logging.info(f"Final response (dict): {final_response}")

        elif isinstance(response, str):
            # If response is a string, use it directly
            final_response = response.strip()
            logging.info(f"Final response (str): {final_response}")

        elif isinstance(response, list):
            # If response is a list, concatenate all string elements
            final_response = ''.join([str(chunk) for chunk in response if isinstance(chunk, (str, dict))])
            logging.info(f"Final response (list): {final_response}")

        else:
            logging.error(f"Unexpected response type: {type(response)}")
            final_response = "Unexpected response type received."

        # Raise an error if no valid response was constructed
        if not final_response:
            raise ValueError("No valid response received from Ollama.")

        logging.info(f"Final response: {final_response}")
        return final_response.strip()

    except Exception as e:
        logging.error(f"Error in submit_prompt: {str(e)}")
        return f"An error occurred: {str(e)}"

def get_table_ddl(conn, table_name):
    """Get the DDL of a table."""
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}';
            """)
            columns = cursor.fetchall()
            ddl = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            ddl += ",\n".join([f"    {col[0]} {col[1]}" for col in columns])
            ddl += "\n);"
            logging.info(f"Fetched DDL for table {table_name}.")
            return ddl
    except psycopg2.Error as e:
        logging.error(f"Database error fetching DDL: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching DDL: {e}")
        return None

def detect_schema_for_table(conn, table_name):
    """Detect the schema where the table exists."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_schema
                FROM information_schema.tables
                WHERE table_name = %s;
            """, (table_name,))
            result = cursor.fetchone()
            if result:
                schema = result[0]
                logging.info(f"Schema detected for table '{table_name}': {schema}")
                return schema  # Return the schema
            else:
                logging.warning(f"No schema found for table '{table_name}'")
                return None
    except Exception as e:
        logging.error(f"Error detecting schema for table '{table_name}': {e}")
        return None
        
def verify_sql_query(query):
    """Temporarily bypass SQL validation for debugging purposes."""
    logging.info("Skipping SQL query validation for debugging.")
    return True

def ingest_sql_into_chromadb(sql_query, question, collection_name="sql_collection"):
    """Ingest a successful SQL query into the specified ChromaDB collection."""
    global client, ef  # Ensure global client and embedding function are accessible

    try:
        # Ensure client and embedding function are initialized
        if not client or not ef:
            logging.error("ChromaDB client or embedding function is not initialized.")
            return

        # Generate a unique document ID based on the SQL query or question
        doc_id = hashlib.md5(sql_query.encode('utf-8')).hexdigest()

        # Get or create the specified collection by name
        collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

        # Check if the document with the same ID already exists in the collection
        existing = collection.get(ids=[doc_id])
        if existing and existing['ids']:
            logging.warning(f"SQL query with ID {doc_id} already exists in {collection_name}. Skipping ingestion.")
            return

        # Generate embedding for the SQL query and the question using Ollama
        response = ollama.embeddings(
            model="bge-m3",  # Ensure the embedding model matches your setup
            prompt=f"SQL Query: {sql_query}\nQuestion: {question}"
        )

        # Extract the embedding vector from the response
        embedding = response.get("embedding")
        if embedding:
            # Ingest the SQL query, question, and embedding into the specified collection
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[json.dumps({
					"sql": sql_query,
					"question": question
				})] # Store both the SQL query and the question as the document
            )
            logging.info(f"SQL query successfully ingested into ChromaDB '{collection_name}' collection.")
        else:
            logging.error(f"Invalid embedding format. Expected a list of floating-point values.")

    except Exception as e:
        logging.error(f"Error ingesting SQL query into ChromaDB {collection_name}: {e}")

def extract_content_from_url(url, extraction_rules):
    """Extracts content from a given URL based on the specified extraction rules."""
    try:
        logging.info(f"Starting content extraction for URL: {url}")

        # Step 1: Fetch the web page content
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Step 2: Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        extracted_content = []

        # Step 3: Apply extraction rules
        for rule in extraction_rules:
            elements = soup.select(rule['selector'])
            for element in elements:
                content = element.get_text(separator='\n').strip()
                if 'filter' in rule and rule['filter']:
                    extracted_content.append(content)

        # Step 4: Post-process the extracted content
        if extracted_content:
            organized_content = organize_extracted_content(extracted_content)
            logging.info(f"Successfully extracted content from URL: {url}")
            return organized_content, None
        else:
            logging.warning(f"No relevant content found at URL: {url}")
            return None, "Failed to extract relevant content from the URL."

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error while fetching URL: {url}. Error: {e}")
        return None, str(e)
    except Exception as e:
        logging.error(f"Exception occurred while extracting content from URL: {url}. Error: {e}")
        return None, str(e)

def organize_extracted_content(content_list):
    """Organizes the extracted content into a well-formatted string."""
    organized_content = "\n\n".join(content_list)
    return organized_content

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file and handle errors."""
    text = ""
    try:
        reader = PdfReader(filepath)
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                logging.warning(f"Text extraction failed for page {page_number + 1}")
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        logging.info(f"Successfully extracted text from PDF: {filepath}")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {filepath}. Error: {e}")
        return None, str(e)
    return text, None

def ingest_text_into_chromadb(text, filename):
    """Ingest text content into the ChromaDB documentation collection."""
    global client, collection  # Ensure global variables are accessible

    try:
        # Generate a document ID based on the filename or URL
        doc_id = hashlib.md5(filename.encode('utf-8')).hexdigest()

        # Check if the document already exists in the collection
        existing = collection.get(ids=[doc_id])
        if existing['ids']:
            logging.warning(f"Document with ID {doc_id} already exists. Skipping ingestion for '{filename}'")
            return

        # Ensure `text` is a single string
        if isinstance(text, list):
            text = "\n".join(text)

        # Generate embedding using Ollama
        response = ollama.embeddings(
            model="bge-m3",  # Replace with the correct embedding model name
            prompt=text
        )

        embedding = response.get("embedding")
        if embedding:
            # Store the text and its embedding in ChromaDB
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text]
            )
            logging.info(f"Document '{filename}' successfully ingested into ChromaDB.")
        else:
            logging.error(f"Failed to generate embedding for document '{filename}'")

    except Exception as e:
        logging.error(f"Error ingesting document into ChromaDB for file: {filename}. Error: {e}")

def get_training_data() -> pd.DataFrame:
    collections = {
        "sql": "sql",
        "ddl_collection": "ddl_collection",
        "documentation": "documentation"
    }

    df_list = []

    for collection_name, training_data_type in collections.items():
        try:
            # Fetch the collection from ChromaDB
            collection = client.get_collection(collection_name)
            data = collection.get()

            # Check if the collection has the expected keys: "documents" and "ids"
            if data and "documents" in data and "ids" in data:
                # Handle SQL collection differently as it might have JSON-encoded documents
                if training_data_type == "sql":
                    documents = [json.loads(doc) if training_data_type == "sql" else doc for doc in data["documents"]]
                else:
                    documents = []
                    for doc in data["documents"]:
                        try:
                            if training_data_type == "sql":
                                documents.append(json.loads(doc))
                            else:
                                documents.append(doc)
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse document in {collection_name}: {doc}")

                # Prepare the DataFrame for each collection
                df = pd.DataFrame({
                    "id": data["ids"],
                    "content": [doc if training_data_type != "sql" else doc.get("sql", "") for doc in documents],
                    "training_data_type": training_data_type
                })

                # For SQL collection, add the 'question' column
                if training_data_type == "sql":
                    df["question"] = [doc.get("question", "") for doc in documents]
                else:
                    df["question"] = None

                # Append the DataFrame to the list
                df_list.append(df)
            else:
                logging.warning(f"No valid data found in collection {collection_name}.")

        except Exception as e:
            logging.error(f"Error fetching data from collection {collection_name}: {e}")

    # Concatenate all DataFrames if data was found in any collection
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df

    # Return an empty DataFrame if no data is found
    return pd.DataFrame()

def remove_training_data(id: str) -> bool:
    collection_suffix_map = {
        "-sql": "sql",
        "-ddl": "ddl_collection",
        "-doc": "documentation"
    }

    for suffix, collection_name in collection_suffix_map.items():
        if id.endswith(suffix):
            client.get_collection(collection_name).delete(ids=id)
            return True
    
    return False

def remove_collection(collection_name: str) -> bool:
    valid_collections = ["sql", "ddl_collection", "documentation"]

    if collection_name in valid_collections:
        client.delete_collection(name=collection_name)
        client.create_collection(name=collection_name, embedding_function=ef)
        return True

    return False

def main():
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'tables';
                """)
                tables = cursor.fetchall()
                all_ddls = []
                for table in tables:
                    table_name = table[0]
                    ddl = get_table_ddl(conn, table_name)
                    if ddl:
                        all_ddls.append(ddl)
                if all_ddls:
                    embed_ddl(all_ddls)
    except Exception as e:
        logging.error(f"Error in main function: {e}")
    finally:
        if conn:
            conn.close()

    # Example usage of chat_with_bot
    question = "What is the current status of the water network?"
    answer = chat_with_bot(question)
    print(answer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ddl', methods=['GET'])
def get_all_ddls():
    try:
        # Step 1: Connect to the PostgreSQL database
        with get_db_connection() as conn:
            if not conn:
                return render_template('error.html', message="Database connection failed"), 500

            # Step 2: Retrieve all table names from the specified schema (e.g., 'tables')
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'tables';
                """)
                tables = cursor.fetchall()

            # Step 3: Generate DDL statements for all tables
            all_ddls = []
            for table in tables:
                table_name = table[0]
                ddl = get_table_ddl(conn, table_name)
                if ddl:
                    all_ddls.append(ddl)

            # Step 4: Embed the DDLs into ChromaDB if there are any
            if all_ddls:
                embed_ddl(all_ddls)

            # Step 5: Render the DDLs in the 'ddl.html' template
            return render_template('ddl.html', ddls=all_ddls)

    except Exception as e:
        # Log any errors and return an error page
        logging.error(f"Error fetching DDLs: {e}")
        return render_template('error.html', message="Error fetching DDLs"), 500

@app.route('/query', methods=['GET'])
def show_query_form():
    return render_template('query.html')

@app.route('/query', methods=['POST'])
def query_database():
    data = request.form
    query = data.get('query')
    question = data.get('question')  # New input for natural language question

    if query:
        try:
            # Check if the query is an SQL statement or a question
            if query.strip().lower().startswith(('select', 'insert', 'update', 'delete')):
                # Execute raw SQL query using the connection pool
                with get_db_connection() as conn:
                    results = query_db(conn, query)
                    if results:
                        return render_template('query.html', results=results)
                    else:
                        return render_template('query.html', error="No results found.")
            else:
                # Assume it's a question for the bot
                bot_response = chat_with_bot(query)
                return render_template('query.html', bot_response=bot_response)
        except psycopg2.OperationalError as e:
            logging.error(f"Database operational error: {e}")
            return render_template('query.html', error="Database connection failed.")
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return render_template('query.html', error="An unexpected error occurred.")
    else:
        return render_template('query.html', error="No query or question provided.")

@app.route('/translate', methods=['GET'])
def show_translate_form():
    return render_template('translate.html')

@app.route('/translate', methods=['POST'])
def translate_and_chat():
    data = request.json
    nl_query = data.get('nl_query')

    if not nl_query:
        return jsonify({'error': 'No natural language query provided.'}), 400

    # Load or initialize the chat history (you can use sessions or persistent storage)
    chat_history = session.get('chat_history', [])

    try:
        # Step 1: Use the new chat_with_bot_and_learn function
        bot_response = chat_with_bot_and_learn(nl_query, chat_history)

        # Step 2: Save the updated chat history back to the session or persistent storage
        session['chat_history'] = chat_history

        # Step 3: Return the response to the user
        return jsonify({'response': bot_response}), 200

    except Exception as e:
        logging.error(f"Error in translate_and_chat: {e}")
        return jsonify({'error': 'An error occurred while processing the query.'}), 500

@app.route('/get_training_data', methods=['GET'])
def get_training_data_route():
    try:
        df = get_training_data()
        if df.empty:
            return render_template('training_data.html', data=None)
        else:
            return render_template('training_data.html', data=df.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Error fetching training data: {e}")
        return render_template('error.html', message="Failed to retrieve training data."), 500

@app.route('/remove_training_data', methods=['POST'])
def remove_training_data_route():
    id = request.form.get('id')
    if not id:
        return jsonify({'error': 'No ID provided.'}), 400

    success = remove_training_data(id)
    if success:
        return jsonify({'message': 'Training data removed successfully.'}), 200
    else:
        return jsonify({'error': 'Failed to remove training data.'}), 500
        
@app.route('/remove_collection', methods=['POST'])
def remove_collection_route():
    collection_name = request.form.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'No collection name provided.'}), 400

    success = remove_collection(collection_name)
    if success:
        return jsonify({'message': f'Collection {collection_name} removed and recreated successfully.'}), 200
    else:
        return jsonify({'error': f'Failed to remove collection {collection_name}.'}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_documentation():
    if request.method == 'POST':
        # Initialize variables for storing the extracted text and messages
        extracted_text, error_message, success_message = None, None, None

        # Check if a file is uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Process the file based on its type
                if filename.lower().endswith('.pdf'):
                    extracted_text, error_message = extract_text_from_pdf(filepath)
                elif filename.lower().endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        extracted_text = f.read()

                if extracted_text and not error_message:
                    # Join extracted text into a single string if it's a list (e.g., PDF extraction)
                    if isinstance(extracted_text, list):
                        extracted_text = "\n".join(extracted_text)
                    ingest_text_into_chromadb(extracted_text, filename)
                    success_message = f"File '{filename}' uploaded, processed, and embedded successfully."
                else:
                    error_message = f"File '{filename}' could not be processed: {error_message}"

                # Clean up the uploaded file
                os.remove(filepath)

        # Check if a URL is provided
        elif request.form.get('url'):
            url = request.form.get('url')
            # Define the extraction rules for the specific type of content you are interested in
            extraction_rules = [
                {"selector": "pre", "filter": lambda content: 'SELECT' in content.upper() or 'CREATE' in content.upper()}
            ]

            extracted_text, error_message = extract_content_from_url(url, extraction_rules)

            if extracted_text and not error_message:
                # Join the extracted content list into a single string
                joined_text = "\n".join(extracted_text)
                ingest_text_into_chromadb(joined_text, url)
                success_message = f"URL '{url}' processed and embedded successfully."
            else:
                error_message = f"URL '{url}' could not be processed: {error_message}"

        else:
            error_message = "No file selected or URL provided."

        return render_template('upload.html', extracted_text=extracted_text, message=success_message or error_message)

    return render_template('upload.html')


@app.route('/error')
def show_error():
    # The error message will be passed as a query parameter
    message = request.args.get('message', 'An unexpected error occurred.')
    return render_template('error.html', message=message)

def handle_error(message):
    # Redirects to the error page with the error message
    return redirect(url_for('show_error', message=message))

# Example usage of handle_error in a route
@app.route('/error', methods=['GET'])
def db_error():
    try:
        # Example database operation using the connection pool
        with get_db_connection() as conn:
            # Perform database operations
            pass
    except psycopg2.OperationalError as e:
        logging.error(f"Database operational error: {e}")
        return handle_error("Database connection failed. Please try again later.")
    except Exception as e:
        logging.error(f"Error during database operation: {e}")
        return handle_error("An unexpected error occurred. Please try again later.")

@app.errorhandler(404)
def page_not_found(e):
    return handle_error("Page not found. Please check the URL and try again."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return handle_error("Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True)
