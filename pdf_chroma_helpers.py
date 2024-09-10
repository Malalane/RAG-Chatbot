import os
import logging
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PDFProcessingError(Exception):
    """Custom exception for errors in PDF processing."""
    pass
import logging
import google.generativeai as genai

def upload_and_initialize_model(config):
    """
    Uploads a PDF file to the Gemini API and initializes a Gemini model based on the provided configuration.

    Parameters:
    - pdf_file: The path to the PDF file to be uploaded.
    - config (dict): A dictionary containing model configuration settings, including:
        - model_name (str): The name of the Gemini model to initialize.
        - safety_settings (dict): Safety settings for the model.
        - generation_config (dict): Generation configuration for the model.

    Returns:
    - model (GenerativeModel): The initialized Gemini model.
    - pdf (UploadedFile): The uploaded PDF file object.
    """
    if not config["pdf_path"] or not isinstance(config["pdf_path"], str):
        logging.error("Invalid image pdf path.")
        raise ValueError("Invalid pdf file path.")
    try:
        # Upload the PDF file
        logging.info("Uploading the PDF file...")
        pdf = genai.upload_file(config["pdf_path"])
        logging.info("PDF file uploaded successfully.")

        # Initialize the Gemini model
        logging.info("Initializing the Gemini model...")
        model = genai.GenerativeModel(
            model_name=config["model_name"],
            safety_settings=config["safety_settings"],
            generation_config=config["generation_config"]
        )
        logging.info("Gemini model initialized successfully.")

        return model,pdf

    except Exception as e:
        logging.error(f"An error occurred during upload or initialization: {e}")
        raise


def load_pdf(file_path: str) -> List[str]:
    """
    Reads the text content from a PDF file, splits it into chunks, and returns a list of text chunks.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[str]: A list of text chunks from the PDF.

    Raises:
        PDFProcessingError: If the file cannot be read or text cannot be extracted.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if not text:
            raise PDFProcessingError("No text could be extracted from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        doc_splits = text_splitter.split_text(text)
        logging.info(f"PDF loaded and split into {len(doc_splits)} chunks.")
        return doc_splits

    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        raise PDFProcessingError(f"Failed to process PDF at {file_path}: {e}")

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function using Gemini AI API for generating document embeddings.

    Args:
        input (Documents): A collection of documents to embed.

    Returns:
        Embeddings: Generated embeddings for the input documents.

    Raises:
        ValueError: If the Gemini API key is not set in environment variables.
    """
    def __init__(self):
        self.gemini_api_key = os.getenv("API_GEMINI")
        if not self.gemini_api_key:
            raise ValueError("Gemini API Key not provided. Set GEMINI_API_KEY as an environment variable.")
        genai.configure(api_key=self.gemini_api_key)

    def __call__(self, input: Documents) -> Embeddings:
        try:
            model_embedding = "models/embedding-001"
            title = "Custom query"
            result = genai.embed_content(
                model=model_embedding,
                content=input,
                task_type="retrieval_document",
                title=title
            )
            logging.info("Embeddings generated successfully.")
            return result["embedding"]

        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

def load_chroma_collection(path: str, name: str):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Args:
        path (str): The path where the Chroma database is stored.
        name (str): The name of the collection within the Chroma database.

    Returns:
        chromadb.Collection: The loaded Chroma Collection.

    Raises:
        ValueError: If the collection cannot be loaded.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=path)
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        logging.info(f"Chroma collection '{name}' loaded successfully from '{path}'.")
        return db
    except Exception as e:
        logging.error(f"Error loading Chroma collection: {e}")
        raise ValueError(f"Failed to load Chroma collection '{name}' at '{path}': {e}")

def create_chroma_db(documents: List[str], path: str, name: str):
    """
    Creates a Chroma database collection using the provided documents or loads an existing one.

    Args:
        documents (List[str]): A list of document texts to be added to the database.
        path (str): The file path where the Chroma database will be stored.
        name (str): The name of the collection within the Chroma database.

    Returns:
        chromadb.Collection: The created or loaded Chroma Collection.

    Raises:
        ValueError: If the collection cannot be created or loaded.
    """
    try:
        # Attempt to load an existing collection
        try:
            db = load_chroma_collection(path, name)
            logging.info(f"Existing collection '{name}' found and loaded.")
        except ValueError:
            # If collection does not exist, create a new one
            chroma_client = chromadb.PersistentClient(path=path)
            db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
            for i, doc in enumerate(documents):
                db.add(documents=doc, ids=str(i))
            logging.info(f"New Chroma DB created with {len(documents)} documents in collection '{name}'.")

        return db

    except Exception as e:
        logging.error(f"Error creating or loading Chroma DB: {e}")
        raise ValueError(f"Failed to create or load Chroma DB at {path}: {e}")
def generate_text_from_pdf(prompt: str,config: dict,model,pdf) -> str:
    """
    Generates text based on an image and a prompt using a Gemini model.

    :param pdf_file: The path to the pdf file.
    :param prompt: The text prompt to guide content generation.
    :param config: dictionary to guide model sesstings.
    :return: Generated text from the model.
    :raises ValueError: If input validation fails.
    :raises RuntimeError: If model interaction fails.
    """

    try:
        # Upload the image file.
        
        if model:
            logging.info("Generating content...")
            response = model.generate_content([prompt, pdf])
            return response.text, model
        else:

            model = upload_and_initialize_model(config)
            # Generate content based on the prompt and image.
            logging.info("Generating content...")
            response = model.generate_content([prompt, pdf])

            logging.info("Content generated successfully.")
            return response.text, model

    except Exception as e:
        logging.error(f"Failed to interact with the GenAI model: {e}")
        raise RuntimeError(f"Model interaction failed: {e}")

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        raise RuntimeError("An unexpected error occurred while generating content.")


def get_relevant_passage(query: str, db, n_results: int) -> str:
    """
    Retrieves the most relevant passage from the Chroma DB based on the query.

    Args:
        query (str): The query text to search for.
        db (chromadb.Collection): The Chroma database collection to query.
        n_results (int): The number of top results to return.

    Returns:
        str: The most relevant passage found in the database.
    """
    try:
        passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
        logging.info(f"Relevant passage retrieved for query: {query}")
        return passage

    except Exception as e:
        logging.error(f"Error retrieving relevant passage: {e}")
        raise ValueError(f"Failed to retrieve relevant passage: {e}")

def make_rag_prompt(query: str, relevant_passage: str) -> str:
    """
    Creates a prompt for RAG (Retrieval-Augmented Generation) using the relevant passage.

    Args:
        query (str): The user's query.
        relevant_passage (str): The relevant passage extracted from the database.

    Returns:
        str: A formatted prompt for the generative model.
    """
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (
        f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and conversational tone. \
If the passage is irrelevant to the answer, you may ignore it.
QUESTION: '{query}'
PASSAGE: '{escaped}'

ANSWER:
"""
    )
    logging.info("RAG prompt created.")
    return prompt

def generate_answer(prompt: str,config: dict) -> str:
    """
    Generates an answer using the Gemini AI generative model based on the provided prompt.

    Args:
        prompt (str): The prompt to provide to the generative model.

    Returns:
        str: The generated answer.
    """
    gemini_api_key = os.getenv("API_GEMINI")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable.")
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name=config["model_name"],safety_settings=config["safety_settings"],
        generation_config=config["generation_config"])
        chat = model.start_chat(history=[])
        answer = chat.send_message(prompt)
        logging.info("Answer generated successfully.")
        return answer.text

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        raise ValueError(f"Failed to generate answer: {e}")
