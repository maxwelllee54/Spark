from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext


conf = SparkConf().setMaster("local[*]").setAppName('HTF_Vol').set('spark.executor.memory', '2g')
sc = SparkContext(conf=conf)
spark = SQLContext(sc)
lst = []

'''
df = spark.read.csv('ctauto.csv', header=True)
print(df.printSchema())
print(df.select(['SYMBOL', 'DATE', 'TIME', 'PRICE','SIZE']).show(n=10))
#print(df.count())

for symbol in df['SYMBOL']:

    lst.append(symbol)

print(lst)
'''