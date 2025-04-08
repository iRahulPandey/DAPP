import pandas as pd
import streamlit as st
import os
import sys
import io
import re
import subprocess
import tempfile
import shutil
import uuid
from pathlib import Path
from langchain_community.llms import Ollama
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser

# Create a custom response parser that captures but doesn't display responses
class CaptureResponseParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
        
    def format_dataframe(self, result):
        # Just return the dataframe without displaying it
        return result["value"]
        
    def format_plot(self, result):
        # Return the path without displaying it
        return result["value"]
        
    def format_other(self, result):
        # Return the text without displaying it
        return result["value"]

# Override os.system to prevent opening external viewers
original_system = os.system
def no_op_system(cmd):
    # Check if the command is trying to open an image
    if "open " in cmd or "xdg-open" in cmd or "start " in cmd:
        # Don't execute the command
        return 0
    # Otherwise, let it run
    return original_system(cmd)

# Apply the override
os.system = no_op_system

# Set page configuration
st.set_page_config(page_title="AI Data Analyst", layout="wide")

# Improve the visual display with custom CSS
st.markdown("""
<style>
    .visualization-container {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .visualization-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #4CAF50;
    }
    .question {
        padding: 10px;
        border-left: 5px solid #0066cc;
        margin-bottom: 10px;
        background-color: #f0f2f6;
    }
    .answer {
        padding: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False

# Helper functions
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        # Skip header line and parse model names from the first column
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]
        return models if models else ["mixtral", "llama3", "gemma"]
    except Exception as e:
        st.error(f"Error fetching Ollama models: {str(e)}")
        return ["mixtral", "llama3", "gemma"]

# Create persistent directory for storing images
image_dir = "saved_plots"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Helper function to create a persistent copy of images
def make_persistent_copy(image_path):
    if not os.path.exists(image_path):
        return None
    
    # Create a unique filename
    unique_filename = f"{image_dir}/plot_{uuid.uuid4()}.png"
    
    # Copy the file
    try:
        shutil.copy2(image_path, unique_filename)
        return unique_filename
    except Exception as e:
        print(f"Error copying image: {e}")
        return image_path

# Callback function to handle input submission
def submit_query():
    if st.session_state.user_input and st.session_state.df is not None:
        query = st.session_state.user_input
        st.session_state.user_input = ""
        st.session_state.processing_query = True
        st.session_state.current_query = query
    elif st.session_state.df is None:
        st.warning("Please upload a CSV file first!")

# Main area setup
st.title("AI Data Analyst")
st.write("Upload a CSV file and ask questions about your data!")

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    available_models = get_ollama_models()
    model = st.selectbox("Select LLM Model", available_models)
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.session_state.raw_df = data
            
            # Create Ollama LLM and use it directly with SmartDataframe
            ollama_llm = Ollama(model=model)
            
            # Create SmartDataframe with the LLM and custom response parser
            st.session_state.df = SmartDataframe(
                data,
                config={
                    "llm": ollama_llm,
                    "response_parser": CaptureResponseParser,
                    "verbose": True,
                    "save_charts": False,
                }
            )
            
            st.success("Data loaded successfully!")
            st.write(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    
    if st.button("Clear Conversation"):
        st.session_state.conversation = []

    if st.session_state.raw_df is not None:
        csv = st.session_state.raw_df.to_csv(index=False)
        st.download_button(
            label="Download Data",
            data=csv,
            file_name="exported_data.csv",
            mime="text/csv"
        )

    if st.session_state.conversation:
        conversation_text = "\n\n".join([f"User: {query}\nAI: {str(response)}\nCode:\n{code}" 
                                       for query, response, code in st.session_state.conversation])
        st.download_button(
            label="Download Conversation",
            data=conversation_text,
            file_name="conversation_history.txt",
            mime="text/plain"
        )

# Display data preview if available
if st.session_state.raw_df is not None:
    with st.expander("Data Preview"):
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
        st.text(f"Total rows: {len(st.session_state.raw_df)}")

# Process query if one is pending
if st.session_state.processing_query and st.session_state.df is not None:
    with st.spinner(f"Thinking about: {st.session_state.current_query}"):
        try:
            # Execute the query
            response = st.session_state.df.chat(st.session_state.current_query)
            generated_code = st.session_state.df.last_code_executed
            
            # Store the result in conversation history
            # For image responses, make a persistent copy
            if isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg')):
                # Create a persistent copy of the image
                persistent_path = make_persistent_copy(response)
                if persistent_path:
                    st.session_state.conversation.append((st.session_state.current_query, persistent_path, generated_code))
                else:
                    st.session_state.conversation.append((st.session_state.current_query, response, generated_code))
            else:
                st.session_state.conversation.append((st.session_state.current_query, response, generated_code))
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.error(f"Details: {type(e).__name__}: {str(e)}")
        
        # Reset processing flag
        st.session_state.processing_query = False
        st.session_state.current_query = None

# Chat interface - display conversation history
for i, (query, response, code) in enumerate(st.session_state.conversation):
    # Display question
    st.markdown(f'<div class="question"><strong>You:</strong> {query}</div>', unsafe_allow_html=True)
    
    # Display answer container start
    st.markdown('<div class="answer">', unsafe_allow_html=True)
    
    # Process different types of responses
    if isinstance(response, pd.DataFrame):
        st.write("<strong>AI:</strong> Here's the result:", unsafe_allow_html=True)
        st.dataframe(response, use_container_width=True)
    elif isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg')):
        # This is a path to an image
        st.write("<strong>AI:</strong> Here's the visualization you requested:", unsafe_allow_html=True)
        try:
            st.image(response)
        except Exception as e:
            st.error(f"Could not display image: {str(e)}")
    elif isinstance(response, dict) and 'plotly' in response:
        st.write("<strong>AI:</strong> Here's the visualization you requested:", unsafe_allow_html=True)
        st.plotly_chart(response['plotly'], use_container_width=True)
    else:
        # Text response
        st.write(f"<strong>AI:</strong> {response}", unsafe_allow_html=True)
    
    # Display answer container end
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Code viewer
    with st.expander(f"View generated code for Query {i+1}"):
        st.code(code, language="python")

# Query input
st.text_input("Ask a question about your data:", key="user_input", on_change=submit_query)

# Example questions
st.markdown("### Example questions you can ask:")
examples = [
    "How many rows are in this dataset?",
    "What are the column names?",
    "Show me a summary of the numerical columns",
    "Create a bar chart of the top 5 values in Category",
    "Show the distribution of values in CustomerAge",
    "Calculate the correlation between Price and Rating",
    "Find any missing values in the dataset",
    "Show me a trend of sales by month",
    "What is the average order value by category?",
    "Create a pie chart showing payment methods"
]
st.write("\n".join([f"- {example}" for example in examples]))