import pandas as pd

data =  pd.read_csv("census.csv")
print len(data)
print len(data[data.income ==  "<=50K"])