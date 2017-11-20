u"""
fetch data from Poloniex Exchange

Created on 14/07/17
by fccoelho
license: GPL V3 or Later
"""

import pandas as pd
import datetime


def get_full_table(pair, start, end):
    """
    Gets at most 300000 raw trades
    """
    df = pd.read_json(
        'https://poloniex.com/public?command=returnTradeHistory&currencyPair={}&start={}&end={}'.format(pair, int(
            start.timestamp()), int(end.timestamp())))
    df.set_index(['date'], inplace=True)
    print('fetched {} {} trades.'.format(df.size, pair))
    return df


def get_price_table(pair, start, end):
    """
    Poloniex API only returns maximum of 300000 trades or 1 year for each pair.
    :returns:
    dictionary with one dataframe per pair
    """
    print('Downloading {} from {} to {}.'.format(pair, start, end))

    df = get_full_table(pair, start, end)

    df = df.resample('1T').mean()  # resample in windows of 1 minute
    df[pair] = df.rate
    for cname in df.columns:
        if cname != pair:
            del df[cname]

    return df


def concatenate_series(rates):
    """
    :parameters:
    - rates: dictionary with the pairs dataframes
    """
    for k, df in rates.items():  # Solve non-unique indices
        rates[k] = df.loc[~df.index.duplicated(keep='first')]
    data = pd.concat(rates, axis=1)
    data.columns = data.columns.droplevel(0)
    print(data.columns)
    data.columns = [name.lower() for name in data.columns]  # convenient to save to PGSQL
    return data


def extend_history(pair, df):
    """
    Extends a dataframe with data on a pair with older data.
    :param pair:
    :param df:
    :return:
    """
    End = df.index.min()
    Start = end - datetime.timedelta(days=364)
    dfextra = get_price_table(pair, Start, End)
    df = df.append(dfextra)  # pd.concat([df,dfextra], axis=0)
    return df

def get_ohlc(pair, start, end):
    """
    Gets OHLC historical data aggregated in 5-minute candlesticks
    :param pair: Currency pair, e.g. USDT_ETH
    :param start: unix timestamp
    :param end: unix timestamp
    :return: dataframe
    """
    print('Downloading {} from {} to {}.'.format(pair, start, end))
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period=300'.format(pair,
                                                                                                          int(start.timestamp()),
                                                                                                          int(end.timestamp()))
    df = pd.read_json(url)
    df['date'] = pd.to_datetime(df.date)
    df.set_index(['date'], inplace=True)
    return df
