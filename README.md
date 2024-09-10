# Chatbot Project

## Overview

This project is a chatbot application that uses an improved retrieval strategy to answer user queries. The chatbot remembers previous questions and retrieves relevant information using a Chroma database. It is hosted using streamlit and assumes you have created a summary of the documentation with your first prompt, create follow up prompts for more engement with the relevant chat. The application uses google gemini as the llm used.

## Project Structure

RAG-Chatbot/ │ ├── streamlit.py # streamlit application ├── pdf_chroma_helpers.py # Helper functions ├── config.json # Configuration file ├── requirements.txt # Dependencies 


## Prerequisites

- Python 3.12 or later
- Virtual environment (venv)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Malalane/RAG-Chatbot.git
   cd RAG-Chatbot
Create a Virtual Environment

Linux/macOS:
bash
Copy code
python3 -m venv myenv
Windows:
cmd
Copy code
python -m venv myenv
Activate the Virtual Environment

Linux/macOS:
bash
Copy code
source myenv/bin/activate
Windows:
cmd
Copy code
myenv\Scripts\activate
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Set Up Configuration
2. **Set config.json for Configuration**
Create a config.json file in the project directory with the following structure:

      ```bash
      {
        "pdf_path": "path/to/your/pdf/file.pdf",
        "chroma_path": "path/to/chroma/database",
        "collection_name": "your_collection_name"
        "safety_settings" : [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}],
        "generation_config" : {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain"
        },
        "model_name":"model/<model_name>"
      }
Replace the placeholders with your actual file paths and collection name.
3. **Set Environment and Environment Variables**

Create API Key for Google Gemini https://ai.google.dev/gemini-api/docs/api-key
after acquiring API Key :

Make sure to set the API_GEMINI environment variable with your Gemini API key.

export API_GEMINI=your_gemini_api_key (or windows equivalent)

4. **Running the Application**
Activate the Virtual Environment (if not already active)
Run this command  on the terminal with while in the folder
Linux/macOS:

source myenv/bin/activate
Windows:
cmd
Copy code
myenv\Scripts\activate


By default, the application will be accessible at http://127.0.0.1:5000.

Usage
Open your browser and navigate to http://127.0.0.1:5000 to access the chatbot interface.
Type your query into the chat input field and press "Send" to receive a response from the chatbot.
Notes
Ensure that your PDF file is accessible and correctly specified in the config.json.
The Chroma database will be created or loaded based on the configuration settings. Ensure you create an empty folder name for the database : mkdir <folder_name> (this will be the path in your config file)
License

Acknowledgements
Streamlit framwork
Google Generative AI for the generative AI API.



