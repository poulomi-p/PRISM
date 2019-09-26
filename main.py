import sys
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 20)
mov = pd.read_csv("datau8.csv")
mov.drop(mov.columns[mov.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
mov.drop(["imdb_id", "homepage", "poster_path", "tagline", "status", "video", "overview"], axis = 1, inplace = True)
print(mov.columns.values)
#print(mov.describe(include = "all"))
print(len(mov))
mov.dropna(subset=["budget"],inplace = True)
mov.dropna(subset=["revenue"],inplace = True)

indexNames = mov[mov['revenue'] == 0].index
mov.drop(indexNames, axis = 0, inplace =True)

indexNames2 = mov[mov['budget'] == 0].index
mov.drop(indexNames2, axis = 0, inplace =True)


perc_profit = pd.Series([])
for index, row in mov.iterrows():
    perc_profit[index] = (mov["revenue"].index - mov["budget"].index)/mov["revenue"].index*100

mov.insert(20,"perc_profit",perc_profit)
numeric_features= ["budget", "popularity", "revenue", "runtime" ,"vote_average", "vote_count","perc_profit"]
print(mov[numeric_features].head())