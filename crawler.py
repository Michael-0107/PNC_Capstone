import queue
from threading import Thread, Lock
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import time
import pandas as pd
import os
import json
import orjson

# ----------------Customized Errors----------------
class CralwerErrors(Exception):
    pass


class NoTableError(CralwerErrors):
    def __init__(self, message):
        self.message = message
        super().__init__()


# ----------------Crawler Class----------------
class FitchCrawler:
    def __init__(self, maxRetry):
        self.queue = queue.Queue()
        self.lock = Lock()
        self.maxRetry = maxRetry
    
    def get_company_name(self, company_ticker):
        """
        Retrieves the company name based on the given stock ticker symbol.

        Args:
            company_ticker (str): The stock ticker symbol of the company.

        Returns:
            str: The name of the company, or an empty string if the name is not found.
        """
        company = yf.Ticker(company_ticker)
        company_name = ""
        if "shortName" in company.info:
            company_name = company.info["shortName"]
        elif "longName" in company.info:
            company_name = company.info["longName"]
        return company_name
    
    def get_fitch_urls(self, company_name):
        """
        Given a company name, this function google it and return urls that match Fitch's rating website address

        Parameters:
            company_name (str): The name of the company to search for.

        Returns:
            list: A list of URLs that match Fitch's rating website address.
        """
        query = f"{company_name} fitch rating"
        google_search_url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response = requests.get(url=google_search_url, headers = headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Regular expression explanation: 
            # [a-zA-Z0-9-]: includes lowercase or uppercase letters, numbers, and hyphen
            # +: mathces one or more occurance of preceding characters
            # \d: number
            # $: end of url
            pattern = re.compile(r"https://www\.fitchratings\.com/entity/[a-zA-Z0-9-]+-\d+$")
            urls = [a['href'] for a in soup.find_all('a', href=True) if pattern.match(a['href'])]
            return urls
        except requests.HTTPError as httperror:
            if response.status_code == 429:
                print("Too Many Requests when getting Fitch rating urls for ", company_name)
            raise


    def fetch_content(self, url, driver):
        """
        Fetches the HTML content of a webpage using Selenium.

        Parameters:
        url (str): The URL of the webpage to fetch.
        driver (selenium.webdriver): An instance of Selenium WebDriver used to interact with the browser.

        Returns:
        str: The HTML content of the webpage.
        """
        driver.get(url)
        time.sleep(3)
        return driver.page_source
    
    def init_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)  # Adjust path if needed
        return driver

    def parse_tables(self, html_content):
        """
        Parses HTML content to extract table data from tables within a specific div.

        Parameters:
        html_content (str): The HTML content to parse.

        Returns:
        list: A list of dictionaries, each containing the headers and rows of a table.
            Each dictionary has two keys:
            - 'headers': a list of column headers.
            - 'rows': a list of rows, where each row is a list of column values.
        """

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
    
    def crawl_website(self, url):
        """
        Crawls a website and extracts table data.

        Parameters:
        url (str): The URL of the website to crawl.

        Returns:
        list: A list of rows, where each row is a list of column values extracted from the tables.
        """

        driver = self.init_driver()
        html_content = self.fetch_content(url, driver)
        driver.quit()

        if html_content:
            tables = self.parse_tables(html_content)
            print(f"Found {len(tables)} tables on {url}:")
            if len(tables) == 0:
                raise NoTableError(f"Find zero table on {url}")
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
            
    def get_ratings_by_quarter(self, date_rating_dict):
        """
        Aggregates ratings by quarter based on the closest available rating before each quarter start.
        Parameters:
        date_rating_dict (dict): A dictionary with dates as keys and ratings as values.
                                The dates should be in the format 'dd-MMM-YYYY'.

        Returns:
        dict: A dictionary with quarters as keys (in the format 'YYYYQX') and ratings as values.

        Example:
        >>> date_rating_dict = {'01-Oct-2009': '4.5', '01-Jan-2010': '4.7', '01-Apr-2010': '4.8'}
        >>> ratings_by_quarter = get_ratings_by_quarter(date_rating_dict)
        >>> print(ratings_by_quarter)
        {'2009Q4': '4.5', '2010Q1': '4.5', '2010Q2': '4.7'}
        """

        df = pd.DataFrame(list(date_rating_dict.items()), columns=['Date', 'Rating'])

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')

        start_date = '2009-10-01'
        end_date = '2024-03-31'
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        df['Quarter'] = df['Date'].dt.to_period('Q')

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
    
    def save_to_json(self, quarter_rating_data, ticker):
        
        json_file_path = './ratings.json'

        if os.path.exists(json_file_path):
            # rb: read-only binary mode
            with open(json_file_path, "rb") as f:
                json_bytes = f.read()

            # Deserialize JSON bytes to Python object
            data = orjson.loads(json_bytes)
        else:
            # Initialize an empty dictionary if the file does not exist
            data = {}

        # Update the data with the new company rating
        data[ticker] = quarter_rating_data

        # Serialize Python object to JSON bytes
        json_bytes = orjson.dumps(data)

        # Save JSON bytes to a file
        with open(json_file_path, "wb") as f:
            f.write(json_bytes)
        
    def get_fitch_rating(self):
        while not self.queue.empty():
            ticker, retried = self.queue.get()
            print("processing: ", (ticker, retried))
            try:
                company_name = self.get_company_name(ticker)
                print("company_name: ", company_name)
                fitch_urls = self.get_fitch_urls(company_name)
                print("Fitch urls: ", fitch_urls)
                if len(fitch_urls) > 0:
                    output = self.crawl_website(fitch_urls[0])
                    dates = output[0][1:]
                    ratings = output[1][1:]
                    ratings_dic = dict(zip(dates,ratings))
                    ratings_by_quarter = self.get_ratings_by_quarter(ratings_dic)
                    with self.lock:
                        self.save_to_json(ratings_by_quarter, ticker)
            except NoTableError as no_table_error:
                print(no_table_error)
            except Exception as e:
                print("Exception occurs: ", e)
                if retried < self.maxRetry:
                    time.sleep(2 ** retried)
                    self.queue.put((ticker, retried + 1))
                else:
                    continue
            finally:
                self.queue.task_done()
    
    def run(self, tickers, num_thread):
        for ticker in tickers:
            self.queue.put((ticker,0))
        
        threads = []
        for _ in range(num_thread):
            thread = Thread(target = self.get_fitch_rating)
            thread.start()
            threads.append(thread)
        
        self.queue.join()

        for t in threads:
            t.join()


if __name__ == "__main__":


    with open("./temp.txt", "r") as file:
        data = file.readlines()
    tickers = [line.split("\t")[1] for line in data]

    cralwer = FitchCrawler(maxRetry = 2)
    cralwer.run(tickers = tickers, num_thread= 5)

        

        

        


                    




        
        

        

