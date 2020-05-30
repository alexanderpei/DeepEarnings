# DeepEarnings

This is a project which attempts to model whether a company will beat or miss earnings announcements based on previous financial data, sentiment analysis, and stock price movement using deep learning. 

### Looking for collaborators / to-do list
Feel free to reach out to me if you're interested in collaborating.
Future ideas:
1. Use LSTMs/RNNs to track OHLCLV + technical indicators before earnings
2. Use advanced NLP methods on 10Q/10K datasets
3. Use reinforcement learning to develop trading strategies
4. Use alternative data (Google trends) 

## Getting Started

This project uses the R package "edgar" to scrape SEC 10Q/10K filings for sentiment analysis. You also need basic things like Pandas, scikit, and numpy. The neural networks will be built using Keras. I'm using PyCharm for everything so just download that and you can clone the repo. Also PyCharm will let you run R scripts. 

You're going to need a text file "list_gvkey_cik.txt" which contains a list of company tickers, GVkeys, and CIK numbers. This list will provide the companies to the get_ZacksEarnings.R to scrape. If you don't plan on doing sentiment analysis, you don't need the CIK numbers. The GVkeys will also be used later for the stock movement data., which I haven't gotten to yet.

## Gathering Data

### Scraping zacks.com earnings

Because I couldn't find access to consensus estimates for earnings easily, I scraped zacks.com for their earnings. For example, data are scraped from this link:
https://www.zacks.com/stock/research/AAPL/earnings-announcements

Run this in your terminal:
```
Rscript get_ZacksEarnings.R
```
This will create a directory "DirtyZacksEarnings" where it stores each earnings for a company in a text file.

### Financial Data

I have access to Compustat North American through my university. I don't think I'm allowed to share that data set because of their terms and conditions. If you do have access yourself, you can get this data set by querying Compustat - Capital IQ / Compustat / North America
- Daily / Fundamentals Quarterly through Wharton Research Data Services. For the variables, select all of them.

### Downloading 10Q/10K files and sentiment analysis (optional)

Again this will require the text file "list_gvkey_cik.txt". I'm using the R package "edgar", which to me as been the easiest way to scrap SEC filings. This package requires a CIK number to scrape filings from. The current sentiment analysis built in to the edgar package is very simple. It just counts the number of positive/negative business words. More advanced NLP methods would obviously be better, but I haven't gotten around to that.
``` 
Rscript get_SecSentiment.R
```

## Cleaning the Data

The data is converted to a Pandas data frame which has dates as the indexes. The time ranges are hardcoded to only be from 01/2008 - 01/2021 so if you're using data before hand or you're a time traveler then you need to change those in the script.

### Clean zacks.com earnings

After you have run the R script which gathers the earnings from zacks.com, you can now run the script to clean these text files into a Pandas data frame:
```
python pp_CleanZacksEarnings.py
```

### Clean compustat data

```
python pp_CleanCompustat.py
```

### Clean sentiment data (optional)
Same thing as above, must have ran the get_SecSentiment.R script before this one.
``` 
python pp_CleanSecSentiment.py
```
