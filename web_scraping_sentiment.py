import time
import requests
from bs4 import BeautifulSoup
import csv
import os

base = 'https://www.sikayetvar.com/vodafone'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_complaint_data(complaint_url):
  
    response = requests.get(complaint_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title
        title_tag = soup.find('h2', class_='complaint-detail-title')
        title = title_tag.text.strip() if title_tag else ""

        # Extract the explanation
        explanation_tag = soup.find('div', class_='complaint-detail-description')
        explanation = explanation_tag.text.strip() if explanation_tag else ""

        return title, explanation
    else:
        print(f'Failed to retrieve complaint page. Status code: {response.status_code}')
        return "", ""

def main():
    """Main function to collect complaint links and extract data."""
    # Create datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Save the CSV file in the datasets directory
    csv_path = os.path.join('datasets', 'sentiment_dataset.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Explanation', 'Target', 'Link'])

        for i in range(1, 51):
            if i == 1:
                base_url = base
            else:
                base_url = f"{base}?page={i}"
                
            response = requests.get(base_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('article')

                for article in articles:
                    title_tag = article.find('h2', class_='complaint-title')
                    if title_tag:
                        # Extract and construct the full URL for the complaint
                        complaint_link = title_tag.find('a')['href']
                        complaint_url = f"https://www.sikayetvar.com{complaint_link}"

                        # Extract data from each complaint page
                        title, explanation = extract_complaint_data(complaint_url)
                        writer.writerow([title, explanation, 0, complaint_url])

                        # Print the result to the console (optional)
                        #print(f'Title: {title}')
                        #print(f'Explanation: {explanation}')
                        #print('---')

                        time.sleep(1)
            else:
                print(f'Failed to retrieve main page. Status code: {response.status_code}')


main()
     