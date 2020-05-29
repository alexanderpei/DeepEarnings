# DeepEarnings

This is a project which attempts to model whether a company will beat or miss earnings announcements based on previous financial data, sentiment analysis, and stock price movement using deep learning. 

## Getting started

This project uses the R package "edgar" to scrape SEC 10Q/10K filings for sentiment analysis. You also need basic things like Pandas, scikit, and numpy. The neural networks will be built using Keras. I'm using PyCharm for everything so just download that and you can clone the repo. Also PyCharm will let you run R scripts. 

## How to run stuff

You're going to need a text file "list_gvkey_cik.txt" which contains a list of company tickers, GVkeys, and CIK numbers. This list will provide the companies to the get_ZacksEarnings.R to scrape. If you don't plan on doing sentiment analysis, you don't need the CIK numbers. The GVkeys will also be used later for the stock movement data. 

### Scraping Zacks Earnings

Because I couldn't find access to consensus estimates for earnings easily, I scraped zacks.com for their earnings. For example, data are scraped from this link:
https://www.zacks.com/stock/research/AAPL/earnings-announcements

Run this in your terminal:
```
Rscript get_ZacksEarnings.R
```
This will create a directory "DirtyZacksEarnings" where it stores each earnings for a company in a text file.

### Downloading 10Q/10K files and Sentiment Analysis (optional)

Again this will require the text file "list_gvkey_cik.txt". I'm using the R package "edgar", which to me as been the easiest way to scrap SEC filings. This package requires a CIK number to scrape filings from.
``` 
Rscript get_SecSentiment.R
```

