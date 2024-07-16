import yfinance as yf
from gdeltdoc import GdeltDoc, Filters
import urllib3
from bs4 import BeautifulSoup
from openai import OpenAI

def get_3_sheet(ticker, period="5y"):

    corp = yf.Ticker(ticker)

    # update data to certain time period
    hist = corp.history(period=period)
    corp.history_metadata

    # income statement
    income_stmt = corp.income_stmt
    # balance sheet
    balance_sheet = corp.balance_sheet
    # cash flow statement
    cashflow = corp.cashflow

    print("Income statement")
    print(income_stmt.head())

    print("-----------")

    print("Balance Sheet")
    print(balance_sheet.head())

    print("-----------")

    print("Cash flow")
    print(cashflow.head())

    return income_stmt, balance_sheet, cashflow

def get_news(company, ticker, startdate, enddate):
    """
    Input:
        company: The name of the company, e.g.: "Nvidia" (Should mind the upper/lower-case in news text)
        ticker: The ticker of the company, e.g.: "NVDA"
        startdate: The start date of the time frame, format: "yyyy-mm-dd", e.g.: "2023-10-24"
        enddate: The start date of the time frame, format: "yyyy-mm-dd", e.g.: "2023-12-25"
    Output:
        fulltext: A list of news with a header of date, e.g.: [ [20231024] Nvidia is doing good.,
                                                                [20231030] NVDA...,
                                                                ....
                                                                ]
    """

    gd = GdeltDoc()

    f = Filters(
        keyword = company,
        start_date = startdate,
        end_date = enddate,
        num_records=50
    )

    articles = gd.article_search(f)

    fulltext = []

    for index, row in articles.iterrows():
        link = row["url"]
        date = row["seendate"]
        if row["language"] != "English": continue
        try:
            http = urllib3.PoolManager()
            response=http.request('GET', link)
            print('Request succeeded: ', response.status)
            soup = BeautifulSoup(response.data, 'html.parser')
        except urllib3.exceptions.HTTPError as e:
            print('Request failed: ', e.reason)
            continue

        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines and short lines
        text = [chunk for chunk in chunks if len(chunk)>200 and (company in chunk or ticker in chunk)]
        # text = '\n'.join(chunk for chunk in chunks if len(chunk)>200)
        for t in text:
            fulltext.append('[{}] {}'.format(date[:8],t))
        
    fulltext.sort(key=lambda x: x[1:9])
    
    return fulltext

def summarize(text):

    """
    Input:
        fulltext: A list of news with a header of date, e.g.: [ [20231024] Nvidia is doing good.,
                                                                [20231030] NVDA...,
                                                                ....
                                                                ]
    Output:
        The text of the summarization
    """    
    ## NOTE ##
    # 1. Please make sure the input token number is not larger than the model token limitation
    #       GPT-4o:         < 128k tokens, input $5 per 1M tokens, output $15 per 1M tokens
    #       GPT-4 Turbo:    < 128k tokens, input $10 per 1M tokens, output $30 per 1M tokens
    #       GPT-3.5 Turbo:  < 16k tokens, input $0.5 per 1M tokens, output $1.5 per 1M tokens
    # 2. Set the API Key before inferencing
    #   API link: https://platform.openai.com/docs/quickstart/account-setup
    #   Tutorial: https://www.datacamp.com/tutorial/gpt4o-api-openai-tutorial

    client = OpenAI(api_key="your_api_key_here")

    MODEL="gpt-3.5-turbo"

    summarization = client.chat.completions.create(
        model=MODEL,
        # Haven't tested yet, please check the results for some companies before inferencing all the companies
        messages=[
            {"role": "system", "content": "Summarize the following information in 100 words, the number in the bracket stands for the date."},
            {"role": "user", "content": "\n".join(text)}
        ]
    )

    print("Assistant: " + summarization.choices[0].message.content)

    return summarization.choices[0].message.content