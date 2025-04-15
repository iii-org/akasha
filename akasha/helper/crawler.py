import requests
from bs4 import BeautifulSoup
from typing import Tuple
from datetime import datetime
import time


def get_text_from_url(url: str) -> Tuple[str, str]:
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Get the title of the web page
        title = soup.title.string if soup.title else "No title found"

        # Remove unwanted elements
        for element in soup(
            [
                "meta",
                "footer",
                "nav",
                "a",
            ]
        ):
            element.decompose()

        # Extract and return the text content
        text_content = soup.get_text(separator=" ", strip=True)
        return title, text_content
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return "", ""


def get_webpage_last_modified(url):
    try:
        # Send a HEAD request to get headers
        response = requests.head(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the Last-Modified header
        last_modified = response.headers.get("Last-Modified")
        if last_modified:
            # Convert to datetime object
            last_modified_date = datetime.strptime(
                last_modified, "%a, %d %b %Y %H:%M:%S %Z"
            )
            # Convert to timestamp (float)
            last_modified_timestamp = time.mktime(last_modified_date.timetuple())
            return last_modified_date, last_modified_timestamp
        else:
            return None
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
