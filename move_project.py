import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import json
import re
from io import StringIO

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.svm import LinearSVC

import operator
io = StringIO()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 25)

# load data
mov = pd.read_csv("datau8_new.csv")

# remove blank columns
mov.drop(mov.columns[mov.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# remove columns with unneeded data
mov.drop(["imdb_id", "homepage", "poster_path", "tagline", "status", "video", "overview"], axis=1, inplace=True)

print('List of Column Titles:')
print(mov.columns.values)
# print(mov.dtypes)
# print(mov.describe(include="all"))
# print('Number of empty values per column:')
# print(mov.isna().sum())
mov.dropna(subset=["budget"], inplace=True)  # drop rows with NaN budgets
mov.dropna(subset=["revenue"], inplace=True)  # drop rows with NaN revenues

# print('Number of empty values after dropping empty budgets and revenues:')
# print(mov.isna().sum())

# Force all numeric data to float64 data type
numeric_features = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "keywords"]
for feature_num in numeric_features:
    mov[feature_num] = mov[feature_num].apply(pd.to_numeric, errors='coerce')

# Remove any where revenue = 0
inv_revenue = mov.loc[mov["revenue"] == 0]
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

# inv_revenue=mov.loc[mov["revenue"]==0]
# print('If zero, empty revenues successfully deleted:')
# print(len(inv_budget))

# inv_budget=mov.loc[mov["budget"]==0]
# print('If zero, empty budgets successfully deleted:')
# print(len(inv_revenue))

# print(mov[numeric_features].head())
# print('New number of movie datapoints:')
# print(len(mov))

# Double check that data is numeric, float64
# print(mov["budget"].dtype)
# print(mov["revenue"].dtype)

# Create percent profit data
mov["perc_profit"] = (mov["revenue"] - mov["budget"]) / mov["budget"] * 100
mov['success'] = mov['revenue'] > mov['budget']
#print(mov['success'])

# Force values to numbers
mov["perc_profit"] = mov["perc_profit"].apply(pd.to_numeric, errors='coerce')

#print(mov["perc_profit"].dtype)
#print(mov['perc_profit'].describe(include='all'))
#print(mov['perc_profit'].isna().sum())
mov.dropna(subset=['perc_profit'], inplace=True)
#print(mov['perc_profit'].isna().sum())

# Display data section with new data
numeric_features = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "perc_profit"]
# print(mov[numeric_features])

# print(mov[numeric_features].corr())

# print(mov[numeric_features].describe(include="all"))

# Manually solved for 10 most popular genres
# drama 2239
# comedy 1667
# thriller 1391
# action 1246
# adventure 888
# romance 881
# crime 789
# scifi 597
# horror 557
# family 506
pop_genres = ["Drama", "Comedy", "Thriller", "Action", "Adventure", "Romance", "Crime", "Science Fiction", "Horror",
              "Family"]

# Find all possible possible genres and their counts

listings = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'cast', 'crew']
max_entries_movies = 3
min_movie = [10, 100, 10, 10, 500, 500] #edit crew members later
lists = ['gen_list', 'producers_list', 'countries_list', 'lang_list', 'cast_list', 'crew_list']
success_list = pd.Series([])
success_dicts = pd.Series([])
top_lists = {}
for i in range(len(listings)):
    list1 = pd.Series([])
    successDict = {}
    list2 = []
    list3 = pd.Series([])
    print(listings[i])
    for index, row in mov.iterrows():
        listString = row[listings[i]]
        l = re.findall('\'name\': \'[0-9a-zA-Z ]*\'', listString)
        for j in range(len(l)):
            st = l[j]
            st = re.sub('name\': \'', '', st)
            st = re.sub('\'', '', st)
            st = re.sub('"', '', st)
            l[j] = st
            if row['success']:
                if st in successDict:
                    successDict[st] += 1
                else:
                    successDict[st] = 1
        list1[index] = l
    mov[lists[i]] = list1
    # Create a list of tuples sorted by index 1 i.e. value field
    listofTuples = sorted(successDict.items(), reverse=True, key=lambda x: x[1])

    # Iterate over the sorted sequence
    for j in range(0, min_movie[i]):
        list2.append(listofTuples[j][0])
    top_lists[lists[i]] = list2

for i in range(len(lists)):
    list_name = lists[i]
    top = top_lists[list_name]
    list_dict = {}
    list_embed = pd.Series([])
    for index, row in mov.iterrows():
        list3 = []
        for j in range(len(row[list_name])):
            entry = row[list_name][j]
            if len(list3) < 3:
                #print(top_lists[i])
                if (entry in top) and not(entry in list3):
                    list3.append(entry)
            else:
                break
        list3 = sorted(list3)
        s = ("".join(list3))
        if s not in list_dict:
            list_dict[s] = len(list_dict)+1
        list_embed[index] = list_dict[s]
        mov.at[index, list_name] = list3
    mov[list_name+'_val'] = list_embed
lists_val = ['gen_list_val', 'producers_list_val', 'countries_list_val', 'lang_list_val', 'cast_list_val', 'crew_list_val']
#print(mov[lists_val].describe(include="all"))


target = ['success']
X_train, X_test, y_train, y_test = train_test_split(mov[lists_val], mov['success'], test_size=0.2, random_state=0)

tfidfer = TfidfTransformer()
X_train_tfidf = tfidfer.fit_transform(X_train)
X_test_tfidf = tfidfer.transform(X_test)

svm = LinearSVC().fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

'''
def reg():
    str_all = ''
    test = []
    str_all = e_genres.get() + " " + e_procom.get() + " " + e_procount.get() + " " + e_cast.get() + " " + e_crew.get() +  " "
    str_all = str_all + " " + e_spl.get()
    test.append(str_all)
    print(X_train.shape)
    X_tr_tfidf = tfidfer.transform(X_train)
    pred = svm.predict(X_tr_tfidf)
    print(pred)
    if pred:
        l_msg['text'] = 'your movie will be successful'
    else:
        l_msg['text'] = 'your movie will not be successful'
'''
root = tk.Tk()
root.title("Drop-down boxes for option selections.")

root.geometry('500x500')
root.title("movie prediction")


l_msg = tk.Label(root, text='INPUT NEW MOVIE INFORMATION')
l_msg.grid(row=0, columnspan=3)
# cast
l_cast1 = tk.Label(root, text='Select Actor 1:')
l_cast1.grid(row=2, sticky=tk.W)
e_cast1 = tk.StringVar(root)
e_cast1.set("")
a1 = ttk.Combobox(root, value=sorted(top_lists['cast_list']), textvariable=e_cast1)
a1.grid(row=2, column=2, sticky=tk.E, padx=3)
# cast 2
l_cast2 = tk.Label(root, text='Select Actor 2:')
l_cast2.grid(row=3, sticky=tk.W)
e_cast2 = tk.StringVar(root)
e_cast2.set("")
a2 = ttk.Combobox(root, value=sorted(top_lists['cast_list']), textvariable=e_cast2)
a2.grid(row=3, column=2, sticky=tk.E, padx=3)
# cast 3
l_cast3 = tk.Label(root, text='Select Actor 3:')
l_cast3.grid(row=4, sticky=tk.W)
e_cast3 = tk.StringVar(root)
e_cast3.set("")
a3 = ttk.Combobox(root, value=sorted(top_lists['cast_list']), textvariable=e_cast3)
a3.grid(row=4, column=2, sticky=tk.E, padx=3)

# crew
l_crew1 = tk.Label(root, text='Select Crew Member 1:')
l_crew1.grid(row=5, sticky=tk.W)
e_crew1 = tk.StringVar(root)
e_crew1.set("")
c1 = ttk.Combobox(root, value=sorted(top_lists['crew_list']), textvariable=e_crew1)
c1.grid(row=5, column=2, sticky=tk.E, padx=3)
# cast 2
l_crew2 = tk.Label(root, text='Select Crew Member 2:')
l_crew2.grid(row=6, sticky=tk.W)
e_crew2 = tk.StringVar(root)
e_crew2.set("")
c2 = ttk.Combobox(root, value=sorted(top_lists['crew_list']), textvariable=e_crew2)
c2.grid(row=6, column=2, sticky=tk.E, padx=3)
# cast 3
l_crew3 = tk.Label(root, text='Select Crew Member 3:')
l_crew3.grid(row=7, sticky=tk.W)
e_crew3 = tk.StringVar(root)
e_crew3.set("")
c3 = ttk.Combobox(root, value=sorted(top_lists['crew_list']), textvariable=e_crew3)
c3.grid(row=7, column=2, sticky=tk.E, padx=3)

# genres
l_genre1 = tk.Label(root, text='Select Genre 1:')
l_genre1.grid(row=8, sticky=tk.W)
e_genre1 = tk.StringVar(root)
e_genre1.set("")
g1 = ttk.Combobox(root, value=sorted(top_lists['gen_list']), textvariable=e_genre1)
g1.grid(row=8, column=2, sticky=tk.E, padx=3)
# genre 2
l_genre2 = tk.Label(root, text='Select Genre 2:')
l_genre2.grid(row=9, sticky=tk.W)
e_genre2 = tk.StringVar(root)
e_genre2.set("")
g2 = ttk.Combobox(root, value=sorted(top_lists['gen_list']), textvariable=e_genre2)
g2.grid(row=9, column=2, sticky=tk.E, padx=3)
# genre 3
l_genre3 = tk.Label(root, text='Select Genre 3:')
l_genre3.grid(row=10, sticky=tk.W)
e_genre3 = tk.StringVar(root)
e_genre3.set("")
g3 = ttk.Combobox(root, value=sorted(top_lists['gen_list']), textvariable=e_genre3)
g3.grid(row=10, column=2, sticky=tk.E, padx=3)

# production_company 1
l_procom1 = tk.Label(root, text='Select Producer 1:')
l_procom1.grid(row=11, sticky=tk.W)
e_procom1 = tk.StringVar(root)
e_procom1.set("")
p1 = ttk.Combobox(root, value=sorted(top_lists['producers_list']), textvariable=e_procom1)
p1.grid(row=11, column=2, sticky=tk.E, padx=3)
# production_company 2
l_procom2 = tk.Label(root, text='Select Producer 2:')
l_procom2.grid(row=12, sticky=tk.W)
e_procom2 = tk.StringVar(root)
e_procom2.set("")
p2 = ttk.Combobox(root, value=sorted(top_lists['producers_list']), textvariable=e_procom2)
p2.grid(row=12, column=2, sticky=tk.E, padx=3)
# production_company 3
l_procom3 = tk.Label(root, text='Select Producer 3:')
l_procom3.grid(row=13, sticky=tk.W)
e_procom3 = tk.StringVar(root)
e_procom3.set("")
p3 = ttk.Combobox(root, value=sorted(top_lists['producers_list']), textvariable=e_procom3)
p3.grid(row=13, column=2, sticky=tk.E, padx=3)


# 'production_countries' 1
l_procount1 = tk.Label(root, text='Select Country 1:')
l_procount1.grid(row=14, sticky=tk.W)
e_procount1 = tk.StringVar(root)
e_procount1.set("")
pr1 = ttk.Combobox(root, value=sorted(top_lists['countries_list']), textvariable=e_procount1)
pr1.grid(row=14, column=2)
# 'production_countries' 2
l_procount2 = tk.Label(root, text='Select Country 2:')
l_procount2.grid(row=15, sticky=tk.W)
e_procount2 = tk.StringVar(root)
e_procount2.set("")
pr2 = ttk.Combobox(root, value=sorted(top_lists['countries_list']), textvariable=e_procount2)
pr2.grid(row=15, column=2)


# 'spoken_languages 1
l_spl1 = tk.Label(root, text='Select Language 1:')
l_spl1.grid(row=16, sticky=tk.W)
e_spl1 = tk.StringVar(root)
e_spl1.set("")
l1 = ttk.Combobox(root, value=sorted(top_lists['lang_list']), textvariable=e_spl1)
l1.grid(row=16, column=2)
# 'spoken_languages 2
l_spl2 = tk.Label(root, text='Select Language 2:')
l_spl2.grid(row=17, sticky=tk.W)
e_spl2 = tk.StringVar(root)
e_spl2.set("")
l2 = ttk.Combobox(root, value=sorted(top_lists['lang_list']), textvariable=e_spl2)
l2.grid(row=17, column=2)


# predict button
f_btn = tk.Frame(root)
b_login = tk.Button(f_btn, text='pred', width=6)
b_login.grid(row=18, column=5)
b_cancel = tk.Button(f_btn, text='cancel', width=6, command=root.quit)
b_cancel.grid(row=18, column=10)
f_btn.grid(row=18, columnspan=2, pady=10)
root.mainloop()
