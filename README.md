# ScrapeMate
Developed using Python and Bright Data's [Scraping Browser](https://brightdata.com/products/scraping-browser), [ScrapeMate](http://scrapemate.streamlit.app) is an intelligent scraping tool that extracts data from any website effortlessly using AI. Built for Researchers, content creators, analysts, and businesses.

## Table of Contents

1. [Introduction](#introduction)
2. [Tech Stack](#tech-stack)
3. [Features](#features)
4. [Quick Start](#quick-start)

## Tech Stack

- Python
- Bright Data
- Streamlit UI
- Selenium
- Groq AI
- BeautifulSoup4
- Pandas

## Features
- Simple, User-Friendly Interface (built with Streamlit UI)
- Dynamic Content Handling (works with JavaScript-loaded pages)
- Infinite Scroll & Pagination Support (handles endless feeds and multi-page content)
- Batch Scraping (scrape multiple URLs at once)
- Accurate and Structured Data Extraction (clean, precise data every time)
- Real-Time Data Scraping (extract live data like stock prices and news updates)
- Custom Field Selection (choose exactly what data you need)
- Fast and Efficient Data Collection (automate data collection and save time)
- Versatile Use Cases (ideal for researchers, developers, marketers, and content creators)
- Data Download Options (download scraped data as CSV or JSON for easy analysis)

## Quick Start

Follow these steps to set up the project locally on your machine.

**Prerequisites**

Make sure you have the following installed on your machine:

- [Git](https://git-scm.com/)
- [Python](https://www.python.org/)

**Cloning the Repository**

```bash
git clone https://github.com/sholajegede/scrapemate.git
cd scrapemate
```

**Create a new folder and Install & activate your virtual environnement**

```bash
python -m venv .venv
```

```bash
.\.venv\scripts\activate
```

**Create a file requirements.txt and copy the following libraries**

```bash
python-dotenv
pandas
pydantic
requests
beautifulsoup4
html2text
langchain 
langchain_ollama
tiktoken
selenium
readability-lxml
streamlit
streamlit-tags
openpyxl
groq
lxml 
html5lib
webdriver-manager
```

**pip install requirements**

```bash
pip install -r requirements.txt 
```

**Set Up Environment Variables**

De-activate your virtual environment and then create a new file named `.env` in the root of your project and add the following:

```env
# BRIGHT DATA
SBR_WEBDRIVER=
```
Re-activate your virtual environment

```bash
.\.venv\scripts\activate
```

**Create assets.py file**
This will include all model names and settings needed for your application to function properly.

```bash
"""
This module contains configuration variables and constants
that are used across different parts of the application.
"""

# Define the pricing for models without Batch API
PRICING = {
    "Groq Llama3.1 70b": {
        "input": 0 ,  # Free
        "output": 0 , # Free
    },
    "Groq Llama3.3 70b": {
       "input": 0 ,  # Free
       "output": 0 , # Free
    },
    # Add other models and their prices here if needed
}

# Timeout settings for web scraping
TIMEOUT_SETTINGS = {
    "page_load": 30,
    "script": 10
}

# Other reusable constants or configuration settings
HEADLESS_OPTIONS = ["--disable-gpu", "--disable-dev-shm-usage","--window-size=1920,1080","--disable-search-engine-choice-screen","--disable-blink-features=AutomationControlled"]


HEADLESS_OPTIONS_DOCKER = ["--headless=new","--no-sandbox","--disable-gpu", "--disable-dev-shm-usage","--disable-software-rasterizer","--disable-setuid-sandbox","--remote-debugging-port=9222","--disable-search-engine-choice-screen"]
#in case you don't need to open the website
##HEADLESS_OPTIONS=HEADLESS_OPTIONS+[ "--headless=new"]

#number of scrolls
NUMBER_SCROLL=2

GROQ_LLAMA_MODEL_FULLNAME="llama-3.1-70b-versatile"
GROQ_LLAMA_MODEL_FULLNAME_OTHER="llama-3.3-70b-versatile"

SYSTEM_MESSAGE = """You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, or extraneous information. 
                        You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON:"""

USER_MESSAGE = f"Extract the following information from the provided text:\nPage content:\n\n"


PROMPT_PAGINATION = """
You are an assistant that extracts pagination elements from markdown content of websites your goal as a universal pagination scrapper of urls from all websites no matter how different they are.

Please extract the following:

- The url of the 'Next', 'More', 'See more', 'load more' or any other button indicating how to access the next page, if any, it should be 1 url and no more, if there are multiple urls with the same structure leave this empty.

- A list of page URLs for pagination it should be a pattern of similar urls with pages that are numbered, if you detect this pattern and the numbers starts from a certain low number until a large number generate the rest of the urls even if they're not included, 
your goal here is to give as many urls for the user to choose from in order for them to do further scraping, you will have to deal with very different websites that can potientially have so many urls of images and other elements, 
detect only the urls that are clearly defining a pattern to show data on multiple pages, sometimes there is only a part of these urls and you have to combine it with the initial url, that will be provided for you at the end of this prompt.

- The user can give you indications on how the pagination works for the specific website at the end of this prompt, if those indications are not empty pay special attention to them as they will directly help you understand the structure and the number of pages to generate.

Provide the output as a JSON object with the following structure:

{
    "page_urls": ["url1", "url2", "url3",...,"urlN"]
}

Do not include any additional text or explanations.
"""
```

**Create scraper.py file**
This is the file responsible for scraping and parsing of data. You would need to setup Bright Data [Scraping Browser](https://brightdata.com/products/scraping-browser) for this part. [See this guide](https://docs.brightdata.com/scraping-automation/scraping-browser/configuration#:~:text=To%20get%20started%2C%20grab%20your,If%20not%2C%20please%20instal%20it.) for setup steps.

```bash
import os
import random
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Type

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken
import streamlit as st

from groq import Groq

from dotenv import load_dotenv
from selenium.webdriver import ChromeOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.remote_connection import RemoteConnection

from api_management import get_api_key
from assets import PRICING,HEADLESS_OPTIONS,SYSTEM_MESSAGE,USER_MESSAGE,GROQ_LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME_OTHER,HEADLESS_OPTIONS_DOCKER
load_dotenv()

def is_running_in_docker():
    """
    Detect if the app is running inside a Docker container.
    This checks if the '/proc/1/cgroup' file contains 'docker'.
    """
    try:
        with open("/proc/1/cgroup", "rt") as file:
            return "docker" in file.read()
    except Exception:
        return False

def setup_selenium(attended_mode=False):
    """
    Set up Selenium WebDriver for Bright Data Scraping Browser (SBR).
    """

    # Define options for Chrome
    options = ChromeOptions()

    # Apply appropriate options based on environment
    if is_running_in_docker():
        for option in HEADLESS_OPTIONS_DOCKER:
            options.add_argument(option)
    else:
        for option in HEADLESS_OPTIONS:
            options.add_argument(option)

    # Fetch Bright Data WebDriver endpoint from environment
    SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER")
    if not SBR_WEBDRIVER:
        raise EnvironmentError("SBR_WEBDRIVER environment variable is not set.")

    try:
        # Connect to Bright Data WebDriver
        print("Connecting to Bright Data Scraping Browser...")
        sbr_connection = RemoteConnection(SBR_WEBDRIVER)
        driver = WebDriver(command_executor=sbr_connection, options=options)
        print("Connected to Bright Data successfully!")
    except Exception as e:
        print(f"Failed to connect to Bright Data Scraping Browser: {e}")
        raise

    return driver

def fetch_html_selenium(url, attended_mode=False, driver=None):
    if driver is None:
        driver = setup_selenium(attended_mode)
        should_quit = True
        if not attended_mode:
            driver.get(url)
    else:
        should_quit = False
        # Do not navigate to the URL if in attended mode and driver is already initialized
        if not attended_mode:
            driver.get(url)

    try:
        if not attended_mode:
            # Add more realistic actions like scrolling
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1);")
            time.sleep(random.uniform(1.1, 1.8))
        # Get the page source from the current page
        html = driver.page_source
        return html
    finally:
        if should_quit:
            driver.quit()

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove headers and footers based on common HTML tags or classes
    for element in soup.find_all(['header', 'footer']):
        element.decompose()  # Remove these tags and their content

    return str(soup)

def html_to_markdown_with_readability(html_content):

    
    cleaned_html = clean_html(html_content)  
    
    # Convert to markdown
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)
    
    return markdown_content
    
def save_raw_data(raw_data: str, output_folder: str, file_name: str):
    """Save raw markdown data to the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, file_name)
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.
    field_name is a list of names of the fields to extract from the markdown.
    """
    # Create field definitions using aliases for Field parameters
    field_definitions = {field: (str, ...) for field in field_names}
    # Dynamically create the model with all field
    return create_model('DynamicListingModel', **field_definitions)


def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.
    """
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))


def trim_to_token_limit(text, model, max_tokens=120000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text

def generate_system_message(listing_model: BaseModel) -> str:
    """
    Dynamically generate a system message based on the fields in the provided listing model.
    """
    # Use the model_json_schema() method to introspect the Pydantic model
    schema_info = listing_model.model_json_schema()

    # Extract field descriptions from the schema
    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        # Get the field type from the schema info
        field_type = field_info["type"]
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    # Create the JSON schema structure for the listings
    schema_structure = ",\n".join(field_descriptions)

    # Generate the system message dynamically
    system_message = f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, or extraneous information. 
                        You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "listings": [
            {{
                {schema_structure}
            }}
        ]
    }} """

    return system_message



def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    token_counts = {}

    if selected_model== "Groq Llama3.1 70b":
        
        # Dynamically generate the system message based on the schema
        sys_message = generate_system_message(DynamicListingModel)
        # print(SYSTEM_MESSAGE)
        # Point to the local server
        client = Groq(api_key=get_api_key("GROQ_API_KEY"),)

        completion = client.chat.completions.create(
        messages=[
            {"role": "system","content": sys_message},
            {"role": "user","content": USER_MESSAGE + data}
        ],
        model=GROQ_LLAMA_MODEL_FULLNAME,
        )

        # Extract the content from the response
        response_content = completion.choices[0].message.content
        
        # Convert the content from JSON string to a Python dictionary
        parsed_response = json.loads(response_content)
        
        # completion.usage
        token_counts = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens
        }

        return parsed_response, token_counts
    elif selected_model== "Groq Llama3.3 70b":
        
        # Dynamically generate the system message based on the schema
        sys_message = generate_system_message(DynamicListingModel)
        # print(SYSTEM_MESSAGE)
        # Point to the local server
        client = Groq(api_key=get_api_key("GROQ_API_KEY"),)

        completion = client.chat.completions.create(
        messages=[
            {"role": "system","content": sys_message},
            {"role": "user","content": USER_MESSAGE + data}
        ],
        model=GROQ_LLAMA_MODEL_FULLNAME_OTHER,
        )

        # Extract the content from the response
        response_content = completion.choices[0].message.content
        
        # Convert the content from JSON string to a Python dictionary
        parsed_response = json.loads(response_content)
        
        # completion.usage
        token_counts = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens
        }

        return parsed_response, token_counts
    else:
        raise ValueError(f"Unsupported model: {selected_model}")


def save_formatted_data(formatted_data, output_folder: str, json_file_name: str, excel_file_name: str):
    """Save formatted data as JSON and Excel in the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse the formatted data if it's a JSON string (from Gemini API)
    if isinstance(formatted_data, str):
        try:
            formatted_data_dict = json.loads(formatted_data)
        except json.JSONDecodeError:
            raise ValueError("The provided formatted data is a string but not valid JSON.")
    else:
        # Handle data from Groq or other sources
        formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Save the formatted data as JSON
    json_output_path = os.path.join(output_folder, json_file_name)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    # Prepare data for DataFrame
    if isinstance(formatted_data_dict, dict):
        # If the data is a dictionary containing lists, assume these lists are records
        data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

    # Create DataFrame
    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")

        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, excel_file_name)
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")
        
        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None

def calculate_price(token_counts, model):
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)
    
    # Calculate the costs
    input_cost = input_token_count * PRICING[model]["input"]
    output_cost = output_token_count * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    
    return input_token_count, output_token_count, total_cost


def generate_unique_folder_name(url):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    url_name = re.sub(r'\W+', '_', url.split('//')[1].split('/')[0])  # Extract domain name and replace non-alphanumeric characters
    return f"{url_name}_{timestamp}"


def scrape_url(url: str, fields: List[str], selected_model: str, output_folder: str, file_number: int, markdown: str):
    """Scrape a single URL and save the results."""
    try:
        # Save raw data
        save_raw_data(markdown, output_folder, f'rawData_{file_number}.md')

        # Create the dynamic listing model
        DynamicListingModel = create_dynamic_listing_model(fields)

        # Create the container model that holds a list of the dynamic listing models
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        # Format data
        formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, selected_model)
        
        # Save formatted data
        save_formatted_data(formatted_data, output_folder, f'sorted_data_{file_number}.json', f'sorted_data_{file_number}.xlsx')

        # Calculate and return token usage and cost
        input_tokens, output_tokens, total_cost = calculate_price(token_counts, selected_model)
        return input_tokens, output_tokens, total_cost, formatted_data

    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return 0, 0, 0, None

# Remove the main execution block if it's not needed for testing purposes
```

**Create pagination_detector.py**

```bash
# pagination_detector.py

import os
import json
from typing import List, Dict, Tuple, Union
from pydantic import BaseModel, Field, ValidationError

import tiktoken
from dotenv import load_dotenv
from groq import Groq

from api_management import get_api_key
from assets import PROMPT_PAGINATION, PRICING, GROQ_LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME_OTHER

load_dotenv()
import logging

class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list, description="List of pagination URLs, including 'Next' button URL if present")

def calculate_pagination_price(token_counts: Dict[str, int], model: str) -> float:
    """
    Calculate the price for pagination based on token counts and the selected model.
    
    Args:
    token_counts (Dict[str, int]): A dictionary containing 'input_tokens' and 'output_tokens'.
    model (str): The name of the selected model.

    Returns:
    float: The total price for the pagination operation.
    """
    input_tokens = token_counts['input_tokens']
    output_tokens = token_counts['output_tokens']
    
    input_price = input_tokens * PRICING[model]['input']
    output_price = output_tokens * PRICING[model]['output']
    
    return input_price + output_price

def detect_pagination_elements(url: str, indications: str, selected_model: str, markdown_content: str) -> Tuple[Union[PaginationData, Dict, str], Dict, float]:
    try:
        """
        Uses AI models to analyze markdown content and extract pagination elements.

        Args:
            selected_model (str): The name of the model to use.
            markdown_content (str): The markdown content to analyze.

        Returns:
            Tuple[PaginationData, Dict, float]: Parsed pagination data, token counts, and pagination price.
        """ 
        prompt_pagination = PROMPT_PAGINATION+"\n The url of the page to extract pagination from   "+url+"if the urls that you find are not complete combine them intelligently in a way that fit the pattern **ALWAYS GIVE A FULL URL**"
        if indications != "":
            prompt_pagination +=PROMPT_PAGINATION+"\n\n these are the users indications that, pay special attention to them: "+indications+"\n\n below are the markdowns of the website: \n\n"
        else:
            prompt_pagination +=PROMPT_PAGINATION+"\n There are no user indications in this case just apply the logic described. \n\n below are the markdowns of the website: \n\n"

        if selected_model == "Groq Llama3.1 70b":
            # Use Groq client
            client = Groq(api_key=get_api_key("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_LLAMA_MODEL_FULLNAME,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
            )
            response_content = response.choices[0].message.content.strip()
            # Try to parse the JSON
            try:
                pagination_data = json.loads(response_content)
            except json.JSONDecodeError:
                pagination_data = {"page_urls": []}
            # Token counts
            token_counts = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            # Calculate the price
            pagination_price = calculate_pagination_price(token_counts, selected_model)

            '''# Ensure the pagination_data is a dictionary
            if isinstance(pagination_data, PaginationData):
                pagination_data = pagination_data.model_dump()
            elif not isinstance(pagination_data, dict):
                pagination_data = {"page_urls": []}'''

            return pagination_data, token_counts, pagination_price

        elif selected_model == "Groq Llama3.3 70b":
            # Use Groq client
            client = Groq(api_key=get_api_key("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_LLAMA_MODEL_FULLNAME_OTHER,
                messages=[
                    {"role": "system", "content": prompt_pagination},
                    {"role": "user", "content": markdown_content},
                ],
            )
            response_content = response.choices[0].message.content.strip()
            # Try to parse the JSON
            try:
                pagination_data = json.loads(response_content)
            except json.JSONDecodeError:
                pagination_data = {"page_urls": []}
            # Token counts
            token_counts = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            # Calculate the price
            pagination_price = calculate_pagination_price(token_counts, selected_model)

            '''# Ensure the pagination_data is a dictionary
            if isinstance(pagination_data, PaginationData):
                pagination_data = pagination_data.model_dump()
            elif not isinstance(pagination_data, dict):
                pagination_data = {"page_urls": []}'''

            return pagination_data, token_counts, pagination_price

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        logging.error(f"An error occurred in detect_pagination_elements: {e}")
        # Return default values if an error occurs
        return PaginationData(page_urls=[]), {"input_tokens": 0, "output_tokens": 0}, 0.0
```

**Create api_management.py**

```bash

import streamlit as st
import os

def get_api_key(api_key_name):
    if api_key_name == 'GROQ_API_KEY':
        return st.session_state['groq_api_key']
    else:
        return os.getenv(api_key_name)
        
```

**Create main.py**
This is where you would build the frontend for the application using Streamlit UI.

```bash
import streamlit as st
from streamlit_tags import st_tags_sidebar
import pandas as pd
import json
from datetime import datetime
from scraper import (
    fetch_html_selenium,
    save_raw_data,
    format_data,
    save_formatted_data,
    calculate_price,
    html_to_markdown_with_readability,
    create_dynamic_listing_model,
    create_listings_container_model,
    scrape_url,
    setup_selenium,
    generate_unique_folder_name
)
from pagination_detector import detect_pagination_elements
import re
from urllib.parse import urlparse
from assets import PRICING
import os

# Initialize Streamlit app
st.set_page_config(
    page_title="ScrapeMate | Your AI partner in web data discovery",
    page_icon="🤌"
)

# Inject Open Graph metadata
st.markdown(
    """
    <meta property="og:title" content="ScrapeMate | Your AI partner in web data discovery">
    <meta property="og:description" content="ScrapeMate is an intelligent scraping tool that extracts data from any website effortlessly.">
    <meta property="og:image" content="https://raw.githubusercontent.com/sholajegede/scrapemate/main/screenshot-3.png">
    <meta property="og:url" content="https://scrapemate.streamlit.app">
    """,
    unsafe_allow_html=True
)

# App content
st.title("ScrapeMate - an intelligent scraping tool that extracts data from any website effortlessly.")

# Initialize session state variables
if 'scraping_state' not in st.session_state:
    st.session_state['scraping_state'] = 'idle'  # Possible states: 'idle', 'waiting', 'scraping', 'completed'
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'driver' not in st.session_state:
    st.session_state['driver'] = None

# Sidebar components
st.sidebar.title("Settings")

# API Keys
with st.sidebar.expander("Add your API key", expanded=False):
    st.session_state['groq_api_key'] = st.text_input("Groq API Key", type="password")

# Model selection
model_selection = st.sidebar.selectbox("Select Model", options=list(PRICING.keys()), index=0)

# URL input
url_input = st.sidebar.text_input("Enter URL(s) separated by whitespace")
# Process URLs
urls = url_input.strip().split()
num_urls = len(urls)
# Fields to extract
show_tags = st.sidebar.toggle("Enable Scraping")
fields = []
if show_tags:
    fields = st_tags_sidebar(
        label='Enter Fields to Extract:',
        text='Press enter to add a field',
        value=[],
        suggestions=[],
        maxtags=-1,
        key='fields_input'
    )

st.sidebar.markdown("---")

# Conditionally display Pagination and Attended Mode options
if num_urls <= 1:
    # Pagination settings
    use_pagination = st.sidebar.toggle("Enable Pagination")
    pagination_details = ""
    if use_pagination:
        pagination_details = st.sidebar.text_input(
            "Enter Pagination Details (optional)",
            help="Describe how to navigate through pages (e.g., 'Next' button class, URL pattern)"
        )

    st.sidebar.markdown("---")

    # Attended mode toggle
    attended_mode = st.sidebar.toggle("Enable Attended Mode")
else:
    # Multiple URLs entered; disable Pagination and Attended Mode
    use_pagination = False
    attended_mode = False
    # Inform the user
    st.sidebar.info("Pagination and Attended Mode are disabled when multiple URLs are entered.")

st.sidebar.markdown("---")



# Main action button
if st.sidebar.button("LAUNCH SCRAPEMATE", type="primary"):
    if url_input.strip() == "":
        st.error("Please enter at least one URL.")
    elif show_tags and len(fields) == 0:
        st.error("Please enter at least one field to extract.")
    else:
        # Set up scraping parameters in session state
        st.session_state['urls'] = url_input.strip().split()
        st.session_state['fields'] = fields
        st.session_state['model_selection'] = model_selection
        st.session_state['attended_mode'] = attended_mode
        st.session_state['use_pagination'] = use_pagination
        st.session_state['pagination_details'] = pagination_details
        st.session_state['scraping_state'] = 'waiting' if attended_mode else 'scraping'

# Scraping logic
if st.session_state['scraping_state'] == 'waiting':
    # Attended mode: set up driver and wait for user interaction
    if st.session_state['driver'] is None:
        st.session_state['driver'] = setup_selenium(attended_mode=True)
        st.session_state['driver'].get(st.session_state['urls'][0])
        st.write("Perform any required actions in the browser window that opened.")
        st.write("Navigate to the page you want to scrape.")
        st.write("When ready, click the 'Resume Scraping' button.")
    else:
        st.write("Browser window is already open. Perform your actions and click 'Resume Scraping'.")

    if st.button("Resume Scraping"):
        st.session_state['scraping_state'] = 'scraping'
        st.rerun()

elif st.session_state['scraping_state'] == 'scraping':
    with st.spinner('Scraping in progress...'):
        # Perform scraping
        output_folder = os.path.join('output', generate_unique_folder_name(st.session_state['urls'][0]))
        os.makedirs(output_folder, exist_ok=True)

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        all_data = []
        pagination_info = None

        driver = st.session_state.get('driver', None)
        if st.session_state['attended_mode'] and driver is not None:
            # Attended mode: scrape the current page without navigating
            # Fetch HTML from the current page
            raw_html = fetch_html_selenium(st.session_state['urls'][0], attended_mode=True, driver=driver)
            markdown = html_to_markdown_with_readability(raw_html)
            save_raw_data(markdown, output_folder, f'rawData_1.md')

            current_url = driver.current_url  # Use the current URL for logging and saving purposes

            # Detect pagination if enabled
            if st.session_state['use_pagination']:
                pagination_data, token_counts, pagination_price = detect_pagination_elements(
                    current_url, st.session_state['pagination_details'], st.session_state['model_selection'], markdown
                )
                # Check if pagination_data is a dict or a model with 'page_urls' attribute
                if isinstance(pagination_data, dict):
                    page_urls = pagination_data.get("page_urls", [])
                else:
                    page_urls = pagination_data.page_urls
                
                pagination_info = {
                    "page_urls": page_urls,
                    "token_counts": token_counts,
                    "price": pagination_price
                }
            # Scrape data if fields are specified
            if show_tags:
                # Create dynamic models
                DynamicListingModel = create_dynamic_listing_model(st.session_state['fields'])
                DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
                # Format data
                formatted_data, token_counts = format_data(
                    markdown, DynamicListingsContainer, DynamicListingModel, st.session_state['model_selection']
                )
                input_tokens, output_tokens, cost = calculate_price(token_counts, st.session_state['model_selection'])
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost
                # Save formatted data
                df = save_formatted_data(formatted_data, output_folder, f'sorted_data_1.json', f'sorted_data_1.xlsx')
                all_data.append(formatted_data)
        else:
            # Non-attended mode or driver not available
            for i, url in enumerate(st.session_state['urls'], start=1):
                # Fetch HTML
                raw_html = fetch_html_selenium(url, attended_mode=False)
                markdown = html_to_markdown_with_readability(raw_html)
                save_raw_data(markdown, output_folder, f'rawData_{i}.md')

                # Detect pagination if enabled and only for the first URL
                if st.session_state['use_pagination'] and i == 1:
                    pagination_data, token_counts, pagination_price = detect_pagination_elements(
                        url, st.session_state['pagination_details'], st.session_state['model_selection'], markdown
                    )
                    # Check if pagination_data is a dict or a model with 'page_urls' attribute
                    if isinstance(pagination_data, dict):
                        page_urls = pagination_data.get("page_urls", [])
                    else:
                        page_urls = pagination_data.page_urls
                    
                    pagination_info = {
                        "page_urls": page_urls,
                        "token_counts": token_counts,
                        "price": pagination_price
                    }
                # Scrape data if fields are specified
                if show_tags:
                    # Create dynamic models
                    DynamicListingModel = create_dynamic_listing_model(st.session_state['fields'])
                    DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
                    # Format data
                    formatted_data, token_counts = format_data(
                        markdown, DynamicListingsContainer, DynamicListingModel, st.session_state['model_selection']
                    )
                    input_tokens, output_tokens, cost = calculate_price(token_counts, st.session_state['model_selection'])
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_cost += cost
                    # Save formatted data
                    df = save_formatted_data(formatted_data, output_folder, f'sorted_data_{i}.json', f'sorted_data_{i}.xlsx')
                    all_data.append(formatted_data)

        # Clean up driver if used
        if driver:
            driver.quit()
            st.session_state['driver'] = None

        # Save results
        st.session_state['results'] = {
            'data': all_data,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_cost': total_cost,
            'output_folder': output_folder,
            'pagination_info': pagination_info
        }
        st.session_state['scraping_state'] = 'completed'
# Display results
if st.session_state['scraping_state'] == 'completed' and st.session_state['results']:
    results = st.session_state['results']
    all_data = results['data']
    total_input_tokens = results['input_tokens']
    total_output_tokens = results['output_tokens']
    total_cost = results['total_cost']
    output_folder = results['output_folder']
    pagination_info = results['pagination_info']

    # Display scraping details
    if show_tags:
        st.subheader("Scraping Results")
        for i, data in enumerate(all_data, start=1):
            st.write(f"Data from URL {i}:")
            
            # Handle string data (convert to dict if it's JSON)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    st.error(f"Failed to parse data as JSON for URL {i}")
                    continue
            
            if isinstance(data, dict):
                if 'listings' in data and isinstance(data['listings'], list):
                    df = pd.DataFrame(data['listings'])
                else:
                    # If 'listings' is not in the dict or not a list, use the entire dict
                    df = pd.DataFrame([data])
            elif hasattr(data, 'listings') and isinstance(data.listings, list):
                # Handle the case where data is a Pydantic model
                listings = [item.dict() for item in data.listings]
                df = pd.DataFrame(listings)
            else:
                st.error(f"Unexpected data format for URL {i}")
                continue
            # Display the dataframe
            st.dataframe(df, use_container_width=True)

        # Display token usage and cost
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Scraping Details")
        st.sidebar.markdown("#### Token Usage")
        st.sidebar.markdown(f"*Input Tokens:* {total_input_tokens}")
        st.sidebar.markdown(f"*Output Tokens:* {total_output_tokens}")
        st.sidebar.markdown(f"**Total Cost:** :green-background[**${total_cost:.4f}**]")

        # Download options
        st.subheader("Download Extracted Data")
        col1, col2 = st.columns(2)
        with col1:
            json_data = json.dumps(all_data, default=lambda o: o.dict() if hasattr(o, 'dict') else str(o), indent=4)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="scraped_data.json"
            )
        with col2:
            # Convert all data to a single DataFrame
            all_listings = []
            for data in all_data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                if isinstance(data, dict) and 'listings' in data:
                    all_listings.extend(data['listings'])
                elif hasattr(data, 'listings'):
                    all_listings.extend([item.dict() for item in data.listings])
                else:
                    all_listings.append(data)
            
            combined_df = pd.DataFrame(all_listings)
            st.download_button(
                "Download CSV",
                data=combined_df.to_csv(index=False),
                file_name="scraped_data.csv"
            )

        st.success(f"Scraping completed. Results saved in {output_folder}")

    # Display pagination info
    if pagination_info:
        st.markdown("---")
        st.subheader("Pagination Information")

        # Display token usage and cost using metrics
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Pagination Details")
        st.sidebar.markdown(f"**Number of Page URLs:** {len(pagination_info['page_urls'])}")
        st.sidebar.markdown("#### Pagination Token Usage")
        st.sidebar.markdown(f"*Input Tokens:* {pagination_info['token_counts']['input_tokens']}")
        st.sidebar.markdown(f"*Output Tokens:* {pagination_info['token_counts']['output_tokens']}")
        st.sidebar.markdown(f"**Pagination Cost:** :blue-background[**${pagination_info['price']:.4f}**]")


        # Display page URLs in a table
        st.write("**Page URLs:**")
        # Make URLs clickable
        pagination_df = pd.DataFrame(pagination_info["page_urls"], columns=["Page URLs"])
        
        st.dataframe(
            pagination_df,
            column_config={
                "Page URLs": st.column_config.LinkColumn("Page URLs")
            },use_container_width=True
        )

        # Download pagination URLs
        st.subheader("Download Pagination URLs")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Pagination CSV",data=pagination_df.to_csv(index=False),file_name="pagination_urls.csv")
        with col2:
            st.download_button("Download Pagination JSON",data=json.dumps(pagination_info['page_urls'], indent=4),file_name="pagination_urls.json")
    # Reset scraping state
    if st.sidebar.button("Clear Results"):
        st.session_state['scraping_state'] = 'idle'
        st.session_state['results'] = None

   # If both scraping and pagination were performed, show totals under the pagination table
    if show_tags and pagination_info:
        st.markdown("---")
        total_input_tokens_combined = total_input_tokens + pagination_info['token_counts']['input_tokens']
        total_output_tokens_combined = total_output_tokens + pagination_info['token_counts']['output_tokens']
        total_combined_cost = total_cost + pagination_info['price']
        st.markdown("### Total Counts and Cost (Including Pagination)")
        st.markdown(f"**Total Input Tokens:** {total_input_tokens_combined}")
        st.markdown(f"**Total Output Tokens:** {total_output_tokens_combined}")
        st.markdown(f"**Total Combined Cost:** :rainbow-background[**${total_combined_cost:.4f}**]")
# Helper function to generate unique folder names
def generate_unique_folder_name(url):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the domain name
    domain = parsed_url.netloc or parsed_url.path.split('/')[0]
    
    # Remove 'www.' if present
    domain = re.sub(r'^www\.', '', domain)
    
    # Remove any non-alphanumeric characters and replace with underscores
    clean_domain = re.sub(r'\W+', '_', domain)
    
    return f"{clean_domain}_{timestamp}"
```

**Run main.py**

```bash
streamlit run main.py
```
This will open your application to [http://localhost:8501](http://localhost:8501) in your browser to view the project.

Start Scraping🚀
