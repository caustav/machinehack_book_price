# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

data_train = pd.read_excel("Data_Train.xlsx")
data_test = pd.read_excel("Data_Test.xlsx")

X = data_train.iloc[:, :-1].to_numpy()
y = data_train.iloc[:, -1].to_numpy()

X = X.astype(np.str)

def extract_bindtype_year_from_edition(data):
    train_edition_col = list(data.iloc[:, 2])
    item_list = list(train_edition_col)
    bindtype = []
    year = []
    
    for i in tqdm(range(len(item_list))):
        try:
            bindtype.append(item_list[i].split("{}".format(','))[0].lower())
            year.append(item_list[i].split("{}".format(','))[-1].lower().replace('-', '').strip().split("{}".format(' '))[-1])
        except:
            pass

    return bindtype, year

def extract_review(data):
    train_review_col = list(data.iloc[:, 3])
    item_list = list(train_review_col)
    review = []
    
    for i in tqdm(range(len(item_list))):
        try:
            review.append(item_list[i].split("{}".format(' '))[0].replace(',', ''))
        except:
            pass
            
    return review

def extract_rating(data):
    train_rating_col = list(data.iloc[:, 4])
    item_list = list(train_rating_col)
    rating = []
    
    for i in tqdm(range(len(item_list))):
        try:
            rating.append(item_list[i].split("{}".format(' '))[0].lower())
        except:
            pass
    
    return rating

bindtype, year = extract_bindtype_year_from_edition(data_train)
review = extract_review(data_train)
rating = extract_rating(data_train)
title = X[:, 0]
author = X[:, 1]
genre = X[:, 6]
category = X[:, 7]

X_main = np.matrix(title).transpose()
X_main = np.append(X_main, np.matrix(author).transpose(), axis=1)
X_main = np.append(X_main, np.matrix(bindtype).transpose(), axis=1)
X_main = np.append(X_main, np.matrix(year).transpose(), axis=1)
X_main = np.append(X_main, np.matrix(review).transpose(), axis=1)
X_main = np.append(X_main, np.matrix(rating).transpose(), axis=1)
X_main = np.append(X_main, np.matrix(genre).transpose(), axis=1)
X_main = np.append(X_main, np.matrix(category).transpose(), axis=1)

X_main[:, 3] = np.char.replace(X_main[:,3], 'print', '')
X_main[:, 3] = np.char.replace(X_main[:,3], ' ', '')
X_main[:, 5] = np.char.replace(X_main[:,5], ',', '')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_0 = LabelEncoder().fit(X_main[:, 0])
X_main[:, 0] = np.matrix(labelencoder_0.transform(X_main[:, 0])).transpose()

labelencoder_1 = LabelEncoder().fit(X_main[:, 1])
X_main[:, 1] = np.matrix(labelencoder_1.transform(X_main[:, 1])).transpose()

labelencoder_2 = LabelEncoder().fit(X_main[:, 2])
X_main[:, 2] = np.matrix(labelencoder_2.transform(X_main[:, 2])).transpose()

labelencoder_3 = LabelEncoder().fit(X_main[:, 3])
X_main[:, 3] = np.matrix(labelencoder_3.transform(X_main[:, 3])).transpose()

labelencoder_6 = LabelEncoder().fit(X_main[:, 6])
X_main[:, 6] = np.matrix(labelencoder_6.transform(X_main[:, 6])).transpose()

labelencoder_7 = LabelEncoder().fit(X_main[:, 7])
X_main[:, 7] = np.matrix(labelencoder_7.transform(X_main[:, 7])).transpose()

from sklearn.preprocessing import OneHotEncoder 

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 6, 7]) 
X_main = onehotencoder.fit_transform(X_main).toarray() 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_main, y, test_size = 0.2, random_state=123)

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

score = 1 - (np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean() ** 0.5)
print("RMLSE XGBRegressor: ", score)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

score = 1 - (np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean() ** 0.5)
print("RMLSE RandomForestRegressor: ", score)


# Trying with Test data

X_t_main = data_test.to_numpy()
X_t_main = X_t_main.astype(np.str)
_bindtype, _year = extract_bindtype_year_from_edition(data_test)
_review = extract_review(data_test)
_rating = extract_rating(data_test)
_title = X_t_main[:, 0]
_author = X_t_main[:, 1]
_genre = X_t_main[:, 6]
_category = X_t_main[:, 7]

X_t = np.matrix(_title).transpose()
X_t = np.append(X_t, np.matrix(_author).transpose(), axis=1)
X_t = np.append(X_t, np.matrix(_bindtype).transpose(), axis=1)
X_t = np.append(X_t, np.matrix(_year).transpose(), axis=1)
X_t = np.append(X_t, np.matrix(_review).transpose(), axis=1)
X_t = np.append(X_t, np.matrix(_rating).transpose(), axis=1)
X_t = np.append(X_t, np.matrix(_genre).transpose(), axis=1)
X_t = np.append(X_t, np.matrix(_category).transpose(), axis=1)

X_t[:, 3] = np.char.replace(X_t[:,3], 'print', '')
X_t[:, 3] = np.char.replace(X_t[:,3], ' ', '')
X_t[:, 5] = np.char.replace(X_t[:,5], ',', '')

labelencoder_0 = LabelEncoder().fit(X_t[:, 0])
X_t[:, 0] = np.matrix(labelencoder_0.transform(X_t[:, 0])).transpose()

labelencoder_1 = LabelEncoder().fit(X_t[:, 1])
X_t[:, 1] = np.matrix(labelencoder_1.transform(X_t[:, 1])).transpose()

labelencoder_2 = LabelEncoder().fit(X_t[:, 2])
X_t[:, 2] = np.matrix(labelencoder_2.transform(X_t[:, 2])).transpose()

labelencoder_3 = LabelEncoder().fit(X_t[:, 3])
X_t[:, 3] = np.matrix(labelencoder_3.transform(X_t[:, 3])).transpose()

labelencoder_6 = LabelEncoder().fit(X_t[:, 6])
X_t[:, 6] = np.matrix(labelencoder_6.transform(X_t[:, 6])).transpose()

labelencoder_7 = LabelEncoder().fit(X_t[:, 7])
X_t[:, 7] = np.matrix(labelencoder_7.transform(X_t[:, 7])).transpose()

from sklearn.preprocessing import OneHotEncoder 

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 6, 7]) 
X_t = onehotencoder.fit_transform(X_t).toarray() 

train_row, train_col = X_main.shape
test_row, test_col = X_t.shape

zeros = np.zeros((test_row, (train_col - test_col)))
X_t = np.append(X_t, zeros, axis=1)

y_pred = regressor.predict(X_t)

solution = pd.DataFrame(y_pred, columns = ['Price'])
solution.to_excel('Predict_Book_Price_Soln.xlsx', index = False)





