import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk


import re
from io import StringIO


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.svm import LinearSVC


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



# Force all numeric data to float64 data type
numeric_features = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "keywords"]
for feature_num in numeric_features:
    mov[feature_num] = mov[feature_num].apply(pd.to_numeric, errors='coerce')

# Remove any where revenue = 0
indexNames = mov[mov['revenue'] == 0].index
mov.drop(indexNames, axis=0, inplace=True)

# Remove any where budget = 0
indexNames2 = mov[mov['budget'] == 0].index
mov.drop(indexNames2, axis=0, inplace=True)

# Create percent profit data
mov["perc_profit"] = (mov["revenue"] - mov["budget"]) / mov["budget"] * 100
mov['success'] = mov['revenue'] > 2*mov['budget']
# print(mov['success'])

# Force values to numbers
mov["perc_profit"] = mov["perc_profit"].apply(pd.to_numeric, errors='coerce')


mov.dropna(subset=['perc_profit'], inplace=True)

# Display data section with new data
numeric_features = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "perc_profit"]

# Find all possible possible text entries per category and their counts in successful movies

# Column names of textual data
listings = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'cast', 'crew']

min_movie = [10, 100, 10, 10, 250, 100]  # find the most popular x entries per category, respectively
max_list = [3, 3, 2, 2, 3, 3] #minimizes movie string lists to x entries, respectively

# Column names of edited lists
lists = ['gen_list', 'producers_list', 'countries_list', 'lang_list', 'cast_list', 'crew_list']

# Pre-allocate variables
success_list = pd.Series([])
success_dicts = pd.Series([])
top_lists = {}

# For each textual column
for i in range(len(listings)):
    list1 = pd.Series([])
    successDict = {}
    list2 = []
    list3 = pd.Series([])
    print(listings[i])

    # For each movie
    for index, row in mov.iterrows():
        listString = row[listings[i]]

        # Parse data from file to list of strings
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
        if listofTuples[j][0] != '':
            list2.append(listofTuples[j][0])
    top_lists[lists[i]] = list2  # Created list of top x entries of category

dicts = {}

# Column names of embedded values
lists_val = ['gen_list_val', 'producers_list_val', 'countries_list_val', 'lang_list_val', 'cast_list_val', 'crew_list_val']

# For each category of textual data
for i in range(len(lists)):
    list_name = lists[i]
    print(lists_val[i])
    top = top_lists[list_name]
    list_dict = {}
    list_embed = pd.Series([])
    x = 1
    list_dict[""] = x

    # Create embedding dictionary of all possible combinations of top values per category
    for j in range(len(top)):
        x += 1
        ind1 = top[j]
        lst = ind1
        list_dict[lst] = x
        for k in range(j + 1, len(top)):
            ind2 = top[k]
            if ind1 == ind2:
                continue
            x += 1
            lst1 = ind1 + ind2
            lst2 = ind2 + ind1
            list_dict[lst1] = x
            list_dict[lst2] = x
            if list_name != 'countries_list' and list_name != 'lang_list':
                for l in range(k + 1, len(top)):
                    ind3 = top[l]
                    if ind1 == ind3 or ind2 == ind3:
                        continue
                    x += 1
                    lst1 = ind1 + ind2 + ind3
                    lst2 = ind1 + ind3 + ind2
                    lst3 = ind2 + ind1 + ind3
                    lst4 = ind2 + ind3 + ind1
                    lst5 = ind3 + ind1 + ind2
                    lst6 = ind3 + ind2 + ind1
                    list_dict[lst1] = x
                    list_dict[lst2] = x
                    list_dict[lst3] = x
                    list_dict[lst4] = x
                    list_dict[lst5] = x
                    list_dict[lst6] = x

    # Limit each's movie's category list to top 3 entries and embed
    for index, row in mov.iterrows():
        list3 = []
        for j in range(len(row[list_name])):
            entry = row[list_name][j]
            if len(list3) < max_list[i]:
                # print(top_lists[i])
                if (entry in top) and not (entry in list3):
                    list3.append(entry)
            else:
                break
        s = ("".join(list3))
        if s not in list_dict:
            x += 1
            list_dict[s] = x
        list_embed[index] = list_dict[s]
        mov.at[index, list_name] = list3
        dicts[list_name] = list_dict
    mov[lists_val[i]] = list_embed

print(mov['success'].describe(include="all"))

# Select features, targets, and split
target = ['success']
X_train, X_test, y_train, y_test = train_test_split(mov[lists_val], mov['success'], test_size=0.2, random_state=0)

# fit model
tfidfer = TfidfTransformer()
X_train_tfidf = tfidfer.fit_transform(X_train)
X_test_tfidf = tfidfer.transform(X_test)


svm = LinearSVC().fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# GUI Button method
def reg():
    X_user = pd.DataFrame([])
    X_act_val = pd.Series([])
    X_crew_val = pd.Series([])
    X_gen_val = pd.Series([])
    X_producers_val = pd.Series([])
    X_countries_val = pd.Series([])
    X_lang_val = pd.Series([])

    X_act_val[0] = dicts['cast_list'][act1.get() + act2.get() + act3.get()]
    X_user['cast_list_val'] = X_act_val
    X_crew_val[0] = dicts['crew_list'][crew1.get() + crew2.get() + crew3.get()]
    X_user['crew_list_val'] = X_crew_val
    X_gen_val[0] = dicts['gen_list'][genre1.get() + genre2.get() + genre3.get()]
    X_user['gen_list_val'] = X_gen_val
    X_producers_val[0] = dicts['producers_list'][procom1.get() + procom2.get() + procom3.get()]
    X_user['producers_list_val'] = X_producers_val
    X_countries_val[0] = dicts['countries_list'][pr1.get() + pr2.get()]
    X_user['countries_list_val'] = X_countries_val
    X_lang_val[0] = dicts['lang_list'][l1.get() + l2.get()]
    X_user['lang_list_val'] = X_lang_val

    X_tr_tfidf_u = tfidfer.transform(X_user)
    pred = svm.predict(X_tr_tfidf_u)
    print(pred)
    if pred:
        l_msg['text'] = 'your movie will be successful'
    else:
        l_msg['text'] = 'your movie will not be successful'

# GUI build
root = tk.Tk()
root.title("Drop-down boxes for option selections.")

root.geometry('500x500')
root.title("movie prediction")

l_msg = tk.Label(root, text='INPUT NEW MOVIE INFORMATION')
l_msg.grid(row=0, columnspan=3)

# cast
l_cast1 = tk.Label(root, text='Select Actor 1:')
l_cast1.grid(row=2, sticky=tk.W)
act1 = tk.StringVar(root)
act1.set("")
a1 = ttk.Combobox(root, value=sorted(top_lists['cast_list']), textvariable=act1)
a1.grid(row=2, column=2, sticky=tk.E, padx=3)
# cast 2
l_cast2 = tk.Label(root, text='Select Actor 2:')
l_cast2.grid(row=3, sticky=tk.W)
act2 = tk.StringVar(root)
act2.set("")
a2 = ttk.Combobox(root, value=sorted(top_lists['cast_list']), textvariable=act2)
a2.grid(row=3, column=2, sticky=tk.E, padx=3)
# cast 3
l_cast3 = tk.Label(root, text='Select Actor 3:')
l_cast3.grid(row=4, sticky=tk.W)
act3 = tk.StringVar(root)
act3.set("")
a3 = ttk.Combobox(root, value=sorted(top_lists['cast_list']), textvariable=act3)
a3.grid(row=4, column=2, sticky=tk.E, padx=3)

# crew
l_crew1 = tk.Label(root, text='Select Crew Member 1:')
l_crew1.grid(row=5, sticky=tk.W)
crew1 = tk.StringVar(root)
crew1.set("")
c1 = ttk.Combobox(root, value=sorted(top_lists['crew_list']), textvariable=crew1)
c1.grid(row=5, column=2, sticky=tk.E, padx=3)
# cast 2
l_crew2 = tk.Label(root, text='Select Crew Member 2:')
l_crew2.grid(row=6, sticky=tk.W)
crew2 = tk.StringVar(root)
crew2.set("")
c2 = ttk.Combobox(root, value=sorted(top_lists['crew_list']), textvariable=crew2)
c2.grid(row=6, column=2, sticky=tk.E, padx=3)
# cast 3
l_crew3 = tk.Label(root, text='Select Crew Member 3:')
l_crew3.grid(row=7, sticky=tk.W)
crew3 = tk.StringVar(root)
crew3.set("")
c3 = ttk.Combobox(root, value=sorted(top_lists['crew_list']), textvariable=crew3)
c3.grid(row=7, column=2, sticky=tk.E, padx=3)

# genres
l_genre1 = tk.Label(root, text='Select Genre 1:')
l_genre1.grid(row=8, sticky=tk.W)
genre1 = tk.StringVar(root)
genre1.set("")
g1 = ttk.Combobox(root, value=sorted(top_lists['gen_list']), textvariable=genre1)
g1.grid(row=8, column=2, sticky=tk.E, padx=3)
# genre 2
l_genre2 = tk.Label(root, text='Select Genre 2:')
l_genre2.grid(row=9, sticky=tk.W)
genre2 = tk.StringVar(root)
genre2.set("")
g2 = ttk.Combobox(root, value=sorted(top_lists['gen_list']), textvariable=genre2)
g2.grid(row=9, column=2, sticky=tk.E, padx=3)
# genre 3
l_genre3 = tk.Label(root, text='Select Genre 3:')
l_genre3.grid(row=10, sticky=tk.W)
genre3 = tk.StringVar(root)
genre3.set("")
g3 = ttk.Combobox(root, value=sorted(top_lists['gen_list']), textvariable=genre3)
g3.grid(row=10, column=2, sticky=tk.E, padx=3)

# production_company 1
l_procom1 = tk.Label(root, text='Select Producer 1:')
l_procom1.grid(row=11, sticky=tk.W)
procom1 = tk.StringVar(root)
procom1.set("")
p1 = ttk.Combobox(root, value=sorted(top_lists['producers_list']), textvariable=procom1)
p1.grid(row=11, column=2, sticky=tk.E, padx=3)
# production_company 2
l_procom2 = tk.Label(root, text='Select Producer 2:')
l_procom2.grid(row=12, sticky=tk.W)
procom2 = tk.StringVar(root)
procom2.set("")
p2 = ttk.Combobox(root, value=sorted(top_lists['producers_list']), textvariable=procom2)
p2.grid(row=12, column=2, sticky=tk.E, padx=3)
# production_company 3
l_procom3 = tk.Label(root, text='Select Producer 3:')
l_procom3.grid(row=13, sticky=tk.W)
procom3 = tk.StringVar(root)
procom3.set("")
p3 = ttk.Combobox(root, value=sorted(top_lists['producers_list']), textvariable=procom3)
p3.grid(row=13, column=2, sticky=tk.E, padx=3)

# 'production_countries' 1
l_procount1 = tk.Label(root, text='Select Country 1:')
l_procount1.grid(row=14, sticky=tk.W)
procount1 = tk.StringVar(root)
procount1.set("")
pr1 = ttk.Combobox(root, value=sorted(top_lists['countries_list']), textvariable=procount1)
pr1.grid(row=14, column=2)
# 'production_countries' 2
l_procount2 = tk.Label(root, text='Select Country 2:')
l_procount2.grid(row=15, sticky=tk.W)
procount2 = tk.StringVar(root)
procount2.set("")
pr2 = ttk.Combobox(root, value=sorted(top_lists['countries_list']), textvariable=procount2)
pr2.grid(row=15, column=2)

# 'spoken_languages 1
l_spl1 = tk.Label(root, text='Select Language 1:')
l_spl1.grid(row=16, sticky=tk.W)
l1 = tk.StringVar(root)
l1.set("")
lang1 = ttk.Combobox(root, value=sorted(top_lists['lang_list']), textvariable=l1)
lang1.grid(row=16, column=2)
# 'spoken_languages 2
l_spl2 = tk.Label(root, text='Select Language 2:')
l_spl2.grid(row=17, sticky=tk.W)
l2 = tk.StringVar(root)
l2.set("")
lang2 = ttk.Combobox(root, value=sorted(top_lists['lang_list']), textvariable=l2)
lang2.grid(row=17, column=2)

# predict button
f_btn = tk.Frame(root)
b_login = tk.Button(f_btn, text='pred', width=6, command=reg)
b_login.grid(row=18, column=5)
b_cancel = tk.Button(f_btn, text='cancel', width=6, command=root.quit)
b_cancel.grid(row=18, column=10)
f_btn.grid(row=18, columnspan=2, pady=10)
root.mainloop()
