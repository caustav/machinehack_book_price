# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(X_train, y_train)

# score_dt = regressor.score(X_test, y_test)
# print("DT Regressor : " + str(score_dt))

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)

# from sklearn.ensemble import GradientBoostingRegressor
# gb_clf = GradientBoostingRegressor(n_estimators=3, learning_rate=1, max_features=2, max_depth=2, random_state=0)
# gb_clf.fit(X_train, y_train)
# y_pred_gb = gb_clf.predict(X_test)
# score_gb = gb_clf.score(X_test, y_test)
# print (score_gb)

# from sklearn.linear_model import LinearRegression

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred_linear = regressor.predict(X_test)

# score = regressor.score(X_test, y_test)
# print (score)

#from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor()
#regressor.fit(X_train, y_train)
#y_pred_dt = regressor.predict(X_test)
#
#score_dt = regressor.score(X_test, y_test)
#print (score_dt)

# print (X_main[100, :])

# X_main = np.delete(X_main, 3, 1)
# X_main = np.delete(X_main, 3, 1)
# X_main = np.delete(X_main, 3, 1)
# X_main = np.delete(X_main, 0, 1)

# title_diff = data_train["Title"].unique().size - data_test["Title"].unique().size