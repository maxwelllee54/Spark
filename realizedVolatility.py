from pyspark.sql import SparkSession, functions as f
from pyspark.sql.window import Window

spark = SparkSession.builder.master('local[*]').appName('HTF_Vol').getOrCreate()
spark.conf.set("spark.executor.memory", "2g")

df = spark.read.csv('newaa.csv', header=True)
df = df.withColumn('TIME', f.date_format(df.TIME, 'HH:mm:ss'))

# patition a window to calculcate the u_sequence
w = Window().partitionBy('SYMBOL', 'DATE').orderBy('TIME')

# calculate the u_sequence
df = df.withColumn('PRICE_1', f.lead('PRICE').over(w))
df = df.withColumn('LOG_RETURN', f.log(df.PRICE_1/df.PRICE))

# patition a window to calculate the moving standard deviation
window_period = 100
w2 = Window().partitionBy('SYMBOL', 'DATE').orderBy('TIME').rowsBetween(0, window_period)
df = df.withColumn('VOLATILITY', f.stddev(df['LOG_RETURN']).over(w2))

df.write.csv('result0527.csv')