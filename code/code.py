from spark_init import spark
from pyspark.sql.functions import col,udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import avg,min,max

# COMMAND ----------

df = spark.read.options(header="True").csv("OfficeDataSet.csv")
df.show()

# COMMAND ----------
print("The count of employees, state, Department")
print('Employee_Count:- {}'.format(df.select("employee_id").count()))

# COMMAND ----------

print('Department_Count:- {}'.format(df.select(col("department")).distinct().count()))

print("State_Count:- {}".format(df.select(col("state")).distinct().count()))

# COMMAND ----------

df2=df.select(col("department").alias("Department Names")).distinct()
df2.show()

# COMMAND ----------

df3=df.groupBy(col("department")).count()
df3.show()

# COMMAND ----------

df4=df.groupBy(col("state")).count()
df4.show()

# COMMAND ----------

df5=df.groupBy(col("state"), col("department")).count()
df5.show()

# COMMAND ----------

df6=df.groupBy(col("department")).agg(min("salary").alias("Minimum Salary") , max("salary").alias("Maximum Salary"))
df6.sort(col("Minimum Salary").asc(), col("Maximum Salary").asc()).show()

# COMMAND ----------

average_bonus = df.filter(df.state == "NY").groupBy("state").agg(avg("bonus").alias("avg_bonus")).select("avg_bonus").collect()[0]["avg_bonus"]
df.filter((df.state=="NY") & (df.department == "Finance") & (df.bonus > average_bonus)).show()

# COMMAND ----------

def raise_salary(age,salary):
    if age > 45 :
        return salary+500
    else:
        return salary
TotalSalary = udf(lambda x,y : raise_salary(x,y), IntegerType())
df.withColumn("Increment", TotalSalary(col("age").cast("Integer"), col("salary").cast("Integer"))).show()


# COMMAND ----------

df=df.filter((col("age").cast("Integer")) > 45)
df.show()

# COMMAND ----------

import pandas as pd

import seaborn as sns

import sklearn.cluster as cluster

df = pd.read_csv('OfficeDataSet.csv')
print(df.describe())
print(sns.pairplot(df[['salary', 'age']]))

kmeans = cluster.KMeans(n_clusters = 5 ,init="k-means++")
kmeans = kmeans.fit(df[['salary','age']])
print(kmeans.cluster_centers_)
df['clusters'] = kmeans.labels_
print(df.head())
df['Clusters'] = kmeans.labels_
df['Clusters'].value_counts()

print(sns.scatterplot(x="salary", y="age",hue = 'Clusters',  data=df))
