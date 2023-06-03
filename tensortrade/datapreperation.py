# Dependecies and requirements

import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from urllib.request import Request, urlopen
import re

url = "https://www.cryptodatadownload.com/data/binance"

req = Request(url)
html_page = urlopen(req)

soup = BeautifulSoup(html_page, 'lxml')

links = []
for link in soup.findAll('a'):
    links.append(link.get('href'))

print(links)

daily = {}
hourly = {}
minute = {}
base_url = "https://www.cryptodatadownload.com"
prefix = "Binance_"

for link in links:
    if link.endswith('.csv'):
        parts = link.split('_')
        dlink = base_url+link
        symbol = parts[1]
        tframe = parts[2].split(".")[0]
        if tframe == "d":
            daily[symbol] = dlink
        if tframe == "1h":
            hourly[symbol] = dlink
        if tframe == "minute":
            minute[symbol] = dlink


print(hourly)
#
# df = pd.read_csv(datalinks[0], skiprows=1)
#
# print(df.dtypes)
