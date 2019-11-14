import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import json
import re
from io import StringIO
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
min_movie = [10, 100, 10, 10, 4800, 4800] #edit crew members later
lists = ['gen_list', 'producers_list', 'countries_list', 'lang_list', 'cast_list', 'crew_list']
success_list = pd.Series([])
success_dicts = pd.Series([])
top_lists = {}
for i in range(len(listings)):
    list1 = pd.Series([])
    successDict = {}
    list2 = pd.Series([])
    list3 = []
    print(listings[i])
    for index, row in mov.iterrows():
    #print(row['title'])
        #print(i)
        #print(listings[i])
        listString = row[listings[i]]
        #if listString == "nan":
        #    continue
        #print(listString)
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
        #print(l)
        list1[index] = l
    mov[lists[i]] = list1
    # Create a list of tuples sorted by index 1 i.e. value field
    listofTuples = sorted(successDict.items(), reverse=True, key=lambda x: x[1])

    # Iterate over the sorted sequence
    for j in range(0, min_movie[i]):
        #print(str(listofTuples[j][1]) + "::" + listofTuples[j][0])
        list2[j] = str(listofTuples[j][0])


    top_lists[lists[i]] = list2
    #success_list[i] = listofTuples
    #success_dicts[i] = successDict


#print(top_lists)
#print(mov[lists].describe(include="all"))

for i in range(len(lists)):
    list_name = lists[i]
    top = top_lists[list_name]
    for index, row in mov.iterrows():
        list3 = []
        for j in range(len(row[list_name])):
            entry = row[list_name][j]
            if len(list3) <= 3:
                #print(top_lists[i])
                if entry in top:
                    '''  
                    list3.append(j)
            else:
                break
        print(list3)
        row[i] = list3
print(mov[lists])


    all_genres = {}
    #print(row['genres'])
    stringG = row['genres'].replace('\'', '"')  # Data uses single quotes, but json interp needs double quotes
    #print("stringG: ")
    #print(stringG)
    mov_gen = []
    l = json.loads(stringG)  # create list of dictionaries
    #print("l: ")
    #print(l)
    for index2 in range(len(l)):
        # print(l[index2])
        if l[index2]['name'] in pop_genres:
            mov_gen.append(l[index2]['name'])
        if l[index2]['name'] in all_genres:
            all_genres[l[index2]['name']] = all_genres[l[index2]['name']] + 1
        else:
            all_genres[l[index2]['name']] = 1
    if len(mov_gen) == 0:
        mov_gen = ["Other"]
    gen_list[index] = mov_gen

    #print("stringCast: ")
    stringCast = row['cast']


    mov_cast = re.findall('\'name\': \'[0-9a-zA-Z ]*\',', stringCast)
    #print(cast)
    for i in range(len(mov_cast)):
        #print(st)
        st = mov_cast[i]
        st = re.sub('\'name\': \'', '', st)
        st = re.sub('\',', '', st)
        mov_cast[i] = st
        #print(st)
    #print(mov_cast)
    cast_list[index] = mov_cast

    stringCrew = row['crew']


    #if (row['title'] == "7.1"): continue

    #print(stringCrew)
    mov_crew = re.findall('\'name\': \'[0-9a-zA-Z ]*\',', stringCrew)
    #print(mov_crew)
    for i in range(len(mov_crew)):
        #print(st)
        st = mov_crew[i]
        st = re.sub('\'name\': \'', '', st)
        st = re.sub('\',', '', st)
        mov_crew[i] = st
        #print(st)
    #print(mov_crew)
    crew_list[index] = mov_crew

    stringProducers = row['production_companies']
    mov_producers = re.findall('\'name\': \'[0-9a-zA-Z ]*\',', stringProducers)
    #print(mov_producers)
    for i in range(len(mov_producers)):
        # print(st)
        st = mov_producers[i]
        st = re.sub('\'name\': \'', '', st)
        st = re.sub('\',', '', st)
        mov_producers[i] = st
        # print(st)
    #print(mov_producers)
    producers_list[index] = mov_producers

mov['gen_list'] = gen_list
mov['cast_list'] = cast_list
mov['crew_list'] = crew_list
mov['producers_list'] = producers_list
#print(all_genres)
'''



mov.drop(listings, axis=1, inplace=True)
#print list of words to file
mov[lists+numeric_features].to_csv('processed.csv')
#success_list.to_csv('success_lists.csv')

