from pyspark.sql import SparkSession, functions as f
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
import pylab as py

def RealizeVol(r):
    return np.dot(r, r)


def TwoScaledRealizedVol(r, K=300):
    array = np.array(r)
    N = int(array.size)
    slow = np.zeros(K)
    for i in range(K):
        try:
            inr = np.arange(start=i, stop=N, step=K)
            slow[i] = RealizeVol(array[inr])
        except IndexError:
            continue
    return slow.mean()

# set up the spark and initialize the variables
spark = SparkSession.builder.master('local[*]').appName('HTF_Vol').config('spark.executor.memory', '2g').getOrCreate()

# use spark to read the local csv file
df = spark.read.csv('newaa.csv', header=True)

# create a window in order to shift columns
w = Window().partitionBy(df['SYMBOL']).orderBy(df['PRICE'])

# shift one position for the stock price in order to calculate log return
df = df.select('*', f.lead('PRICE').over(w).alias('PRICE_1')).na.drop()
df = df.select('*', f.log(df.PRICE_1/df.PRICE).alias('LOG_RETURN'))
#df = df.select('*').where(df['LOG_RETURN'] != 0)

# transfer data into pandas dataframe to handle the data by each stock
pd_df = df.toPandas()

# set up the index for pandas dataframe
tuples = list(zip(pd_df.iloc[:, 0], pd_df.iloc[:, 1]))
index = pd.MultiIndex.from_tuples(tuples, names=['SYMBOL', 'DATE'])
pd_df = pd_df.set_index(index)
result = pd.DataFrame(index=index)
indexList = list(set(pd_df.index.values))

# set up the k values and interval value
k = 300
j = 1

for i in indexList:
    array = np.array(pd_df.ix[i,6].astype(float))
    rv = RealizeVol(array)
    tsrv_avg = TwoScaledRealizedVol(array)
    N = int(array.size)
    K = (N - k + 1) / k
    J = (N - j + 1) / j

    tsrv = (1.0 / (1.0 - K/J)) * (tsrv_avg - (K / J) * rv)
    noise = rv / (2 * N)
    result.loc[i, 'RV']  = rv
    result.loc[i, 'TSRV_AVG'] = tsrv_avg
    result.loc[i, 'TSRV'] = tsrv


q = float(result['TSRV'].quantile(0.95))
result = result[result.TSRV > q]

symbolList = result.index.values

print(result)

print(pd_df)
'''
fig1 = py.figure(figsize = (20,10), dpi = 80)
fig1.suptitle('Top Stocks with 95% Volatility', fontsize = 20)
for symbol in symbolList:
    pd_df = pd_df[pd_df.SYMBOL == symbol]
    volPlot = py.subplot(111)
    volPlot.plot(result.iloc[:, 'TSRV'], label=symbol)
    volPlot.legend(loc='upper left')
    py.ylabel('Volatility')
fig1.savefig('Top Stocks with 95% Volatility.png')
'''





