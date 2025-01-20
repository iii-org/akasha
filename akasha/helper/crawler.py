import requests
from bs4 import BeautifulSoup
from typing import Tuple


def get_text_from_url(url: str) -> Tuple[str, str]:
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the title of the web page
        title = soup.title.string if soup.title else 'No title found'

        # Remove unwanted elements
        for element in soup([
                'meta',
                'footer',
                'nav',
                'a',
        ]):
            element.decompose()

        # Extract and return the text content
        text_content = soup.get_text(separator=' ', strip=True)
        return title, text_content
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return "", ""
