import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# import scipy.stats as scs
# import statsmodels.tsa.stattools as stat
import statsmodels.graphics.tsaplots as spl
# import statsmodels.graphics.gofplots
# import statsmodels.tsa.seasonal as season
from statsmodels.tsa.seasonal import seasonal_decompose

# import statsmodels.formula.api as smf
# import statsmodels.tsa.api as smt
# import statsmodels.api as sm
# import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import ProbPlot

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datetime import datetime
from zeep import Client
# from zeep.wsse.username import UsernameToken
import requests
from io import StringIO
# import io
import yfinance as yf

def download_int_rates(from_date = '2021-01-01T00:00:00', to_date = '2025-01-01T23:59:59'):
    ''' 
    Собираем % ставки (Кривая бескупонной доходности)
    '''
    client = Client(wsdl='https://www.cbr.ru/secinfo/secinfo.asmx?WSDL')
    response = client.service.zcyc_params(FromDate=from_date, ToDate=to_date)
    data = []
    for item in response['_value_1']['_value_1']:
        x = item['ZCYC'] # Zero Cupoun Yield Curve
        record = {
            'date': x['D0'],
            'y_0_25': x['v_0_25'], # срок до погашения лет 0.25
            'y_0_5': x['v_0_5'],
            'y_0_75': x['v_0_75'],
            'y_1': x['v_1_0'],
            'y_2': x['v_2_0'],
            'y_3': x['v_3_0'],
            'y_5': x['v_5_0'],
            'y_7': x['v_7_0'],
            'y_10': x['v_10_0'],
            'y_15': x['v_15_0'],
            'y_20': x['v_20_0'],
            'y_30': x['v_30_0'], # 30 лет
        }
        data.append(record)
    return pd.DataFrame(data)

def get_moex_data_bond(security, from_date, to_date):
    '''
    Скачиваем данные по облигациям
    '''
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/{security}.json"
    all_data = []
    start = 0
    limit = 100

    while True:
        params = {
            "from": from_date,
            "till": to_date,
            "marketprice_board": 1,
            "limit": limit,
            "start": start
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extracting the data from JSON
        columns = data['history']['columns']
        rows = data['history']['data']

        if not rows:
            break

        all_data.extend(rows)
        start += limit

    # Creating DataFrame
    df = pd.DataFrame(all_data, columns=columns)

    return df

def get_moex_data_sha(security, from_date, to_date):
    '''
    Скачиваем данные по акциям
    '''
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{security}.json"
    all_data = []
    start = 0
    limit = 100

    while True:
        params = {
            "from": from_date,
            "till": to_date,
            "marketprice_board": 1,
            "limit": limit,
            "start": start
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extracting the data from JSON
        columns = data['history']['columns']
        rows = data['history']['data']

        if not rows:
            break

        all_data.extend(rows)
        start += limit

    # Creating DataFrame
    df = pd.DataFrame(all_data, columns=columns)

    return df

def get_moex_data_idx(security, from_date, to_date):
    '''
    Скачиваем индексы
    '''
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{security}.json"
    all_data = []
    start = 0
    limit = 100

    while True:
        params = {
            "from": from_date,
            "till": to_date,
            "marketprice_board": 1,
            "limit": limit,
            "start": start
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extracting the data from JSON
        columns = data['history']['columns']
        rows = data['history']['data']

        if not rows:
            break

        all_data.extend(rows)
        start += limit

    # Creating DataFrame
    df = pd.DataFrame(all_data, columns=columns)

    return df