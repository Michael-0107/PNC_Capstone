import yfinance as yf
from gdeltdoc import GdeltDoc, Filters
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

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

    if '.' in ticker:
        tckl = ticker.split('.')
        ticker = tckl[0]

    f = Filters(
        keyword = "NYSE:{}".format(ticker),
        start_date = startdate,
        end_date = enddate,
        num_records=50
    )

    articles = gd.article_search(f)
    if len(articles) == 0:
        print("There are no articles about this company.")
        return []

    fulltext = []

    print("Crawling news...")
    for index, row in articles.iterrows():
        link = row["url"]
        date = row["seendate"]
        if row["language"] != "English": continue
        try:
            http = urllib3.PoolManager()
            response=http.request('GET', link)
            # print('Request succeeded: ', response.status)
            soup = BeautifulSoup(response.data, 'html.parser')
        except urllib3.exceptions.HTTPError as e:
            # print('Request failed: ', e.reason)
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
        if ticker:
            text = [chunk for chunk in chunks if len(chunk)>200 and (company in chunk or ticker in chunk)]
        else:
            text = [chunk for chunk in chunks if len(chunk)>200 and company in chunk]
        # text = [chunk for chunk in chunks if len(chunk)>200]
        # text = '\n'.join(chunk for chunk in chunks if len(chunk)>200)
        for t in text:
            fulltext.append('[{}] {}'.format(date[:8],t))

    length = [len(news) for news in fulltext]
    total_length = sum(length)
    while total_length > 40000 or len(fulltext) > 100:
        fulltext.pop()
        total_length -= length.pop()
        
    fulltext.sort(key=lambda x: x[1:9])
    print("Got {} News!".format(len(fulltext)))
    print("Total length: {}".format(total_length))
    
    return fulltext

def summarize_news(pipeline, company, ticker, startdate, enddate):

    news_list = get_news(company, ticker, startdate, enddate)
    if not news_list:
        return "There are no information of this company for this quarter."
    new_all = []

    for news in tqdm(news_list):
        sub_news = pipeline("Please summarize the following news in 30 words.\nNews:{}\nSummary: ".format(news), max_new_tokens=40, do_sample=False, return_full_text=False, pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
        new_all.append(sub_news)

    output = pipeline("You're a finance expert and you are sharing some thoughts based on the news within a quarter. Please read all the news and write a summarization of this company in this quarter.\nNews:{}\nSummary: ".format(new_all), max_new_tokens=128, do_sample=False, return_full_text=False, pad_token_id=pipeline.tokenizer.eos_token_id, eos_token_id=pipeline.tokenizer.eos_token_id)

    last_period = output[0]['generated_text'].rfind('.')
    return output[0]['generated_text'][:last_period+1]