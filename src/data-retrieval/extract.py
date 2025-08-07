import requests
import os
import sys
from bs4 import BeautifulSoup
import zipfile
from io import BytesIO

def download_and_extract_zip(url, download_dir):
    # Get the response from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Extract the filename from the 'Content-Disposition' header if available, or fall back to URL parsing
    content_disposition = response.headers.get('content-disposition')
    if content_disposition:
        filename = content_disposition.split('filename=')[-1].strip('"')
    else:
        filename = os.path.basename(url)

    # If the filename does not end in .zip, ensure it's named appropriately
    if not filename.endswith('.zip'):
        filename += '.zip'

    # Path to save the downloaded zip file
    zip_path = os.path.join(download_dir, filename)

    # Write the zip file to the specified directory
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # Extract the zip file to a folder named after the zip (without .zip extension)
    extract_dir = os.path.join(download_dir, os.path.splitext(filename)[0])
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(extract_dir)

    print(f"Downloaded and extracted {filename} to {extract_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_extract.py <urls_file> <download_directory>")
    else:
        urls_file = sys.argv[1]
        download_dir = sys.argv[2]
        
        # Ensure the download directory exists
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Read URLs from the file
        with open(urls_file, 'r', encoding='utf-8') as file:
            urls = file.readlines()

        for url in urls:
            url = url.strip()
            if url:
                try:
                    download_and_extract_zip(url, download_dir)
                except Exception as e:
                    print(f"Failed to download or extract {url}: {e}")
