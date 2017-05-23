from pyspark.sql import SparkSession, functions as f
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
import pylab as py
import matplotlib.pyplot as plt
import time


def RealizeVol(r):
        return np.dot(r, r)

def TwoScaledRealizedVol(r, K = 300):
    array = np.array(r)
    N = int(array.size)
    print(N)
    slow = np.zeros(K)
    for i in range(K):
        try:
            inr = np.arange(start=i, stop=N, step=K)
            print(inr)
            slow[i] = RealizeVol(array[inr])
            print(slow[i])
        except IndexError:
            continue
    return slow.mean()

def foo(element):
    return ((element[0], element[1]), element[6])


spark = SparkSession.builder.master('local[*]').appName('HTF_Vol').getOrCreate()
spark.conf.set("spark.executor.memory", "2g")

timeStart = time.clock()
df = spark.read.csv('ctauto.csv', header=True)

df.show()
print('Spark takes {:.4f} seconds to process the file'.format(time.clock()-timeStart))

timeStart = time.clock()
w = Window().partitionBy(df['SYMBOL']).orderBy(df['PRICE'])

df = df.select('*', f.lead('PRICE').over(w).alias('PRICE_1'))

df.show()
print('Spark takes {:.4f} seconds to calculate the PRICE_1'.format(time.clock()-timeStart))


df = df.select('*', f.log(df.PRICE_1/df.PRICE).alias('LOG_RETURN'))


#df = df.select('*').where(df['LOG_RETURN'] != 0)
df = df.select('SYMBOL', 'DATE', 'LOG_RETURN').groupBy('SYMBOL', 'DATE').sum('LOG_RETURN')

#df.show()


pd_df = df.toPandas()

pd_df.columns = ['SYMBOL', 'DATE', 'RV']

#pd_df = pd_df.set_index('SYMBOL')

group = pd_df.groupby(['SYMBOL']).mean()

q = float(group.quantile(0.95))
group = group[group.RV > q]

symbolList = group.index.values

print(symbolList)

fig1 = py.figure(figsize = (20,10), dpi = 80)
fig1.suptitle('Top Stocks with 95% Volatility', fontsize = 20)
for symbol in symbolList:
    result = pd_df[pd_df.SYMBOL == symbol]
    volPlot = plt.subplot(111)
    volPlot.bar(np.arange(len(result.iloc[:, 2])), result.iloc[:, 2], 0.3, facecolor='#9999ff', edgecolor='white',
                  label=symbol)
    volPlot.legend(loc='upper left')
    py.xlabel('Date')
    py.ylabel('Volatility')

py.show()






