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
    #"Groq Llama3.3 70b": {
    #    "input": 0 ,  # Free
    #    "output": 0 , # Free
    #},
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