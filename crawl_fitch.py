import time
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
from pprint import pprint
import json
import os
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import urllib.parse
import yfinance as yf




def get_company_name_from_google_search(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    query = f"{ticker} ticker"
    query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        element = soup.select_one(".PZPZlf.ssJ7i.B5dxMb")
        if element:
            return element.text.strip()
        else:
            return "Company name not found."
    else:
        return "Failed to retrieve data."


def google_search(company_name):
    # Perform a Google search query
    search_query = f"{company_name} fitch rating"
    google_search_url = f"https://www.google.com/search?q={search_query}"

    # Set the headers to mimic a browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Send the request to Google
    response = requests.get(google_search_url, headers=headers)
    response.raise_for_status()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all links that match the pattern
    pattern = re.compile(r"https://www.fitchratings.com/entity/.*")
    urls = [a['href'] for a in soup.find_all('a', href=True) if pattern.match(a['href'])]

    return urls


# Function to initialize the Selenium WebDriver
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run headless Chrome
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)  # Adjust path if needed
    return driver

# Function to fetch page content using Selenium
def fetch_content(url, driver):
    driver.get(url)
    time.sleep(2)  # Wait for the page to load (adjust the sleep time if needed)
    return driver.page_source

# Function to parse HTML content and extract table data from tables within a specific div
def parse_tables(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table_div = soup.find('div', class_='table table--2')

    if not table_div:
        print("No div with class 'table table--2' found.")
        return []

    tables = table_div.find_all('table', class_='table__wrapper')

    table_data = []

    for table in tables:
        headers = [header.get_text(strip=True) for header in table.find_all('th')]
        rows = table.find_all('tr')
        table_rows = []

        for row in rows:
            columns = row.find_all('td')
            if columns:
                table_rows.append([column.get_text(strip=True) for column in columns])

        table_data.append({'headers': headers, 'rows': table_rows})

    return table_data

# Function to crawl a website and extract table data
def crawl_website(url):
    driver = init_driver()
    html_content = fetch_content(url, driver)
    driver.quit()

    if html_content:
        tables = parse_tables(html_content)
        print(f"Found {len(tables)} tables with class 'table__wrapper' in div 'table table--2' on {url}:")
        output = []
        for i, table in enumerate(tables, start=1):
            print(f"\nTable {i}:")
            if table['headers']:
                print("Headers:", table['headers'])
            for row in table['rows']:
                output.append(row)
        return output
    else:
        print(f"Failed to retrieve content from {url}")

def get_ratings_dic(output):
    dates = output[0][1:]
    ratings = output[1][1:]
    ratings_dic = dict(zip(dates,ratings))
    return ratings_dic

def get_ratings_by_quarter(date_rating_dict):
    df = pd.DataFrame(list(date_rating_dict.items()), columns=['Date', 'Rating'])

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')

    # Filter the DataFrame for the specified date range
    start_date = '2009-10-01'
    end_date = '2024-03-31'
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Create a new column for quarters
    df['Quarter'] = df['Date'].dt.to_period('Q')

    # Initialize a dictionary to store the closest rating before each quarter
    quarter_rating_dict = {}

    # Get unique quarters in the range
    quarters = pd.period_range(start=start_date, end=end_date, freq='Q')

    # Iterate through each quarter and find the closest date before the start of the quarter
    for quarter in quarters:
        quarter_start_date = quarter.start_time
        closest_dates = df[df['Date'] < quarter_start_date].sort_values(by='Date', ascending=False)
        if not closest_dates.empty:
            closest_date = closest_dates.iloc[0]
            quarter_rating_dict[str(quarter)] = closest_date['Rating']

    # Display the dictionary
    return quarter_rating_dict

# Main execution
if __name__ == "__main__":


    tickers = [
        "ARKO", "ANF", "ASO", "AAP", "AD.AS", "ACI", "ATD", "AMZN", "ABG", "AN",
        "AZO", "BJ", "BBWI", "BBY", "BURL", "CWH", "GOOS", "CTC.A", "CVNA", "TCS",
        "COST", "DKS", "ADDDS", "DG", "DLTR", "DOL", "FND", "FL", "FRG", "GPS",
        "GPC", "GPI", "HD", "IMKTA", "JOAN", "JILL", "KSS", "KR", "LESL", "QRTEA",
        "LAD", "LOW", "M", "CPRI", "MCW", "MUSA", "EYE", "JWN", "ORLY", "PAG",
        "WOOF", "RL", "RH", "ROST", "SAH", "TJX", "TPR", "TGT", "CURV", "TSCO",
        "VVV", "VSCO", "WBA", "WMT"
    ]

    company_urls = {}
    for ticker in tickers:
        company = yf.Ticker(ticker)
        company_name = ""
        # print(company.info)
        if "shortName" in company.info:
            company_name = company.info["shortName"]
        elif "longName" in company.info:
            company_name = company.info["longName"]
        else:
            continue
        
        urls = google_search(company_name)
        if urls:
            company_urls[ticker] = urls[0]

    # sample_urls = {
    # "ARKO": "https://www.fitchratings.com/entity/arcos-dorados-holdings-inc-90585086",
    # "AMZN": "https://www.fitchratings.com/entity/amazoncom-inc-96631782#ratings",
    # "AZO": "https://www.fitchratings.com/entity/autozone-inc-81657867",
    # "GOOS": "https://www.fitchratings.com/entity/golden-goose-spa-96890427",
    # "TCS": "https://www.fitchratings.com/entity/tata-consultancy-services-limited-94065501",
    # "HD": "https://www.fitchratings.com/entity/home-depot-inc-the-80090927#ratings",
    # "KSS": "https://www.fitchratings.com/entity/kohl-corporation-80338524",
    # "QRTEA": "https://www.fitchratings.com/entity/qurate-retail-inc-96747639",
    # "M": "https://www.fitchratings.com/entity/macy-inc-80089239",
    # "WMT": "https://www.fitchratings.com/entity/walmart-inc-80090564"
    # }

    
    for company, url in company_urls.items():
        output = crawl_website(url)
        print("output: ", output)
        date_rating_dict = get_ratings_dic(output)
        quarter_rating_dict = get_ratings_by_quarter(date_rating_dict)
        # Path to the JSON file
        json_file_path = './retail_ratings.json'

        # Check if the file exists
        if os.path.exists(json_file_path):
            # Read the existing data
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            # Initialize an empty dictionary if the file does not exist
            data = {}

        # Update the data with the new company rating
        data[company] = quarter_rating_dict

        # Write the updated data back to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

