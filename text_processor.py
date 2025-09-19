import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#Scraping from website pages
BASE_URL = "https://hushbposervices.com"

PAGES_TO_SCRAPE = [
    "/",
    "/careers/",
    "/contacts/",
    "/about/",
    "/services/",
    "/services/software-development/",
    "/services/back-office-operations/",
    "/services/customer-service/",
    "/services/finance-accounting/",
    "/services/underwriting/",
]
OUTPUT_FILE = "website_data.json"

def scrape_website_content():
    """
    Collects text data from a predefined list of web pages.
    Each page's extracted content is stored as a dictionary.
    Returns a list of these dictionaries, one per document.
    """
    scraped_documents = []
    
    # Define a custom User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Scraping content from: {BASE_URL}")

    for page_path in PAGES_TO_SCRAPE:
        # Join the base URL and the page path
        full_url = urljoin(BASE_URL, page_path)
        print(f"  -> Scraping page: {full_url}")
        
        try:
            # Pass the headers with the request
            response = requests.get(full_url, headers=headers, timeout=10)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract relevant text.
            page_text = soup.get_text(separator=' ', strip=True)
            
            # Create a document dictionary
            doc_id = os.path.basename(page_path.strip('/')) or "home"
            document = {
                "id": doc_id,
                "text": page_text,
                "metadata": {"source": full_url}
            }
            scraped_documents.append(document)
            
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {full_url}: {e}")
            
    return scraped_documents

# Execute the scraping and save to JSON
if __name__ == "__main__":
    content_data = scrape_website_content()
    
    if content_data:
        try:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(content_data, f, indent=4)
            print(f"Successfully scraped content and saved to {OUTPUT_FILE}.")
        except Exception as e:
            print(f"Failed to write to file. Error: {e}")
    else:
        print("No content was scraped. Exiting.")
