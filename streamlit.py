import streamlit as st
from pdf_chroma_helpers import load_pdf, create_chroma_db, load_chroma_collection, get_relevant_passage, make_rag_prompt, generate_answer,generate_text_from_pdf,upload_and_initialize_model
import json
import os
from langchain_core.messages import HumanMessage

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)


def llm_function(query,model):
    response = model.generate_content(query)

    # Displaying the Assistant Message
    
    with st.chat_message("assistant"):
        st.markdown(response.text)

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content": query
        }
    )

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": response.text
        }
    )
def initialize_model(config):
    uploaded_pdf = config["pdf_path"]
    
    if uploaded_pdf:
        # Process the PDF and create Chroma DB
        
        doc_splits = load_pdf(uploaded_pdf)
        db_path = config["chroma_path"]
        collection_name = config["collection_name"]
        
        # Create or load Chroma DB
        if os.path.exists(db_path):
            db = load_chroma_collection(db_path, collection_name)
        else:
            db, _ = create_chroma_db(doc_splits, db_path, collection_name)
        
        st.success("PDF processed and Chroma DB updated!")
    # Process and store Query and Response
    model, pdf = upload_and_initialize_model(config)
    return model, pdf, db

# Streamlit application
def main():
    # Initialize model and vector database
    model, pdf, db = initialize_model(config)
    st.title("Chatbot")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role":"assistant",
                "content":"Ask me Anything"
            }
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Query input
    query = st.chat_input("Enter your query")
    
    if query:
        # Retrieve relevant passage
        passage = get_relevant_passage(query, db, config["n_results"])
        st.write("Document Definition:")
        st.write(passage[0])
        st.write(query)
        if len(passage)==2:
            
            fuller_query = f"use this detail to perform a proffessional and educational summary with interesting points relating the passage. PASSAGE: '{passage[1]}' while taking in consideration of the prompt. PROMPT: '{query}'"
            prompt,model = generate_text_from_pdf(fuller_query,config,model,pdf)
            llm_function(prompt,model)
        else:
        # Generate and display answer
            fuller_query = f"use this detail to perform an proffessional and educational summary with interesting points relating the passage. PASSAGE: '{passage[1]}' while taking in consideration of the prompt. be clear and consise while still giving relevant information PROMPT: '{query}'"
            prompt = make_rag_prompt(fuller_query, relevant_passage="".join(passage))
            #llm_function(prompt,model)
            answer = generate_answer(prompt,config)
            with st.chat_message("assistant"):
                st.markdown(answer)

            # Storing the User Message
            st.session_state.messages.append(
                {
                    "role":"user",
                    "content": query
                }
            )

            # Storing the User Message
            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":answer
                }
            )

if __name__ == "__main__":
    main()
