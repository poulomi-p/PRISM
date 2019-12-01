import numpy as np
import tkinter as tk
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import json
import re
from io import StringIO

import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.svm import LinearSVC
import numpy as np

# import spacy

io = StringIO()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 25)

# load data
mov = pd.read_csv("datau8_new.csv")

# remove blank columns
mov.drop(mov.columns[mov.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# remove columns with unneeded data
mov.drop(["imdb_id", "homepage", "poster_path", "status", "video"], axis=1, inplace=True)

print('List of Column Titles:')
print(mov.columns.values)
# print(mov.dtypes)
# print(mov.describe(include="all"))
# print('Number of empty values per column:')
# print(mov.isna().sum())
mov.dropna(subset=["budget"], inplace=True)  # drop rows with NaN budgets
mov.dropna(subset=["revenue"], inplace=True)  # drop rows with NaN revenues
mov.dropna(subset=["keywords"], inplace=True)
# mov.dropna(subset=["runtime"], inplace=True)
# mov.dropna(subset=["vote_average"], inplace=True)
mov.dropna(subset=["tagline"], inplace=True)
mov.dropna(subset=["overview"], inplace=True)
mov.dropna(subset=["spoken_languages"], inplace=True)
mov.dropna(subset=["cast"], inplace=True)
mov.dropna(subset=["crew"], inplace=True)
# print('Number of empty values after dropping empty budgets and revenues:')
# print(mov.isna().sum())

# Force all numeric data to float64 data type
numeric_features = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]
for feature_num in numeric_features:
    mov[feature_num] = mov[feature_num].apply(pd.to_numeric, errors='coerce')

# Remove any where revenue = 0
inv_revenue = mov.loc[mov['revenue'] == 0]
# print('Number of 0 revenues: ')
# print(len(inv_revenue))
indexNames = mov[mov['revenue'] == 0].index
mov.drop(indexNames, axis=0, inplace=True)

# Remove any where budget = 0
inv_budget = mov.loc[mov["budget"] == 0]
# print('Number of 0 budgets: ')
# print(len(inv_budget))
indexNames2 = mov[mov['budget'] == 0].index
mov.drop(indexNames2, axis=0, inplace=True)

# Create percent profit data
mov["perc_profit"] = (mov["revenue"] - mov["budget"]) / mov["budget"] * 100
mov['success'] = mov['revenue'] > mov['budget']

# Force values to numbers
mov["perc_profit"] = mov["perc_profit"].apply(pd.to_numeric, errors='coerce')

mov.dropna(subset=['perc_profit'], inplace=True)

# Display data section with new data
numeric_features = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "perc_profit"]

# Find all possible possible genres and their counts


listings = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'cast', 'crew', 'keywords',
            'tagline']

for i in range(len(listings)):
    list1 = pd.Series([])

    for index, row in mov.iterrows():
        listString = row[listings[i]]
        str = ''
        l = re.findall('\'name\': \'[0-9a-zA-Z ]*\'', listString)
        for j in range(0, len(l)):
            l[j] = l[j].replace("\'name\': ", "")
            l[j] = l[j].replace("\'", "")
            str = str + " " + l[j]
        list1[index] = str

mov[listings[i]] = list1

# process vote_average
'''
for index,row in mov.iterrows():
    mov['vote_average'][index]=round(row['vote_average'])
    mov['runtime'][index] = round(row['runtime'])
mov['vote_average'] = mov['vote_average'].astype(np.str)
mov['runtime'] = mov['runtime'].astype(np.str)

mov.dropna(subset=["vote_average"], inplace=True)
'''

target = ['success']
text = ['genres', 'production_companies', 'production_countries', 'cast', 'crew', 'title', 'keywords', 'tagline',
        'overview', 'spoken_languages']
X_train, X_test, y_train, y_test = train_test_split(mov[text], mov['success'], test_size=0.2, random_state=0)
# text=['original_title']


counter = CountVectorizer(ngram_range=(1, 3))
# X_train_bow=counter.fit_transform(X_train)
genlist_train = []
genlist_test = []
for index, row in X_train.iterrows():
    str_all = ''
    for lis in range(0, len(text)):
        str_all = str_all + " " + row[text[lis]]
    genlist_train.append(str_all)
X_train_bow = counter.fit_transform(genlist_train)

for index, row in X_test.iterrows():
    str_all = ''
    for lis in range(0, len(text)):
        str_all = str_all + " " + row[text[lis]]
    genlist_test.append(row[text[lis]])
X_test_bow = counter.transform(genlist_test)

tfidfer = TfidfTransformer()
X_train_tfidf = tfidfer.fit_transform(X_train_bow)
X_test_tfidf = tfidfer.transform(X_test_bow)

svm = LinearSVC().fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))


# mnb_tfid = MultinomialNB()
# mnb_tfid.fit(X_train_tfidf, y_train)
# mnb_tfid_y_predict = mnb_tfid.predict(X_test_tfidf)
# print(classification_report(y_test, mnb_tfid_y_predict))

def reg():
    str_all = ''
    test = []
    str_all = e_genres.get() + " " + e_procom.get() + " " + e_procount.get() + " " + e_cast.get() + " " + e_crew.get() + " " + e_title.get()
    str_all = str_all + " " + e_keywords.get() + " " + e_tagline.get() + " " + e_overview.get() + " " + e_spl.get()
    test.append(str_all)
    X_tr_bow = counter.transform(test)
    print(X_tr_bow.shape)
    X_tr_tfidf = tfidfer.transform(X_tr_bow)
    pred = svm.predict(X_tr_tfidf)
    print(pred)
    if pred:
        l_msg['text'] = 'your movie will be successful'
    else:
        l_msg['text'] = 'your movie will not be successful'


root = tk.Tk()
root.geometry('500x500')
root.title("movie prediction")

l_msg = tk.Label(root, text='INPUT NEW MOVIE INFORMATION')
l_msg.grid(row=0, columnspan=3)

# movie_title
l_title = tk.Label(root, text='title')
l_title.grid(row=1, sticky=tk.W)
e_title = tk.Entry(root)
e_title.grid(row=1, column=2, sticky=tk.E, padx=3)

# cast
l_cast = tk.Label(root, text='cast')
l_cast.grid(row=2, sticky=tk.W)
e_cast = tk.Entry(root)
e_cast.grid(row=2, column=2, sticky=tk.E, padx=3)

# crew
l_crew = tk.Label(root, text='crew')
l_crew.grid(row=3, sticky=tk.W)
e_crew = tk.Entry(root)
e_crew.grid(row=3, column=2, sticky=tk.E, padx=3)

# tagline
l_tagline = tk.Label(root, text='tagline')
l_tagline.grid(row=4, sticky=tk.W)
e_tagline = tk.Entry(root)
e_tagline.grid(row=4, column=2, sticky=tk.E, padx=3)

# keywords
l_keywords = tk.Label(root, text='keywords')
l_keywords.grid(row=5, sticky=tk.W)
e_keywords = tk.Entry(root)
e_keywords.grid(row=5, column=2, sticky=tk.E, padx=3)

# genres
l_genres = tk.Label(root, text='genres')
l_genres.grid(row=6, sticky=tk.W)
e_genres = tk.Entry(root)
e_genres.grid(row=6, column=2, sticky=tk.E, padx=3)

# production_companies
l_procom = tk.Label(root, text='production_companies')
l_procom.grid(row=7, sticky=tk.W)
e_procom = tk.Entry(root)
e_procom.grid(row=7, column=2, sticky=tk.E, padx=3)

# 'production_countries'
l_procount = tk.Label(root, text='production_countries')
l_procount.grid(row=8, sticky=tk.W)
e_procount = tk.Entry(root)
e_procount.grid(row=8, column=2, sticky=tk.E, padx=3)

# overview'
l_overview = tk.Label(root, text='overview')
l_overview.grid(row=9, sticky=tk.W)
e_overview = tk.Entry(root)
e_overview.grid(row=9, column=2, sticky=tk.E, padx=3)

# 'spoken_languages
l_spl = tk.Label(root, text='spoken_languages')
l_spl.grid(row=10, sticky=tk.W)
e_spl = tk.Entry(root)
e_spl.grid(row=10, column=2, sticky=tk.E, padx=3)
# predict button
f_btn = tk.Frame(root)
b_login = tk.Button(f_btn, text='pred', width=6, command=reg)
b_login.grid(row=11, column=5)
b_cancel = tk.Button(f_btn, text='cancel', width=6, command=root.quit)
b_cancel.grid(row=11, column=10)
f_btn.grid(row=11, columnspan=2, pady=10)

root.mainloop()
