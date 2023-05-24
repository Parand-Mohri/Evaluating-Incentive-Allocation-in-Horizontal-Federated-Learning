from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def RF_CV_cen_test (X_train, Y_train, X_test, Y_test):
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, Y_train)
    final_model = CV_rfc.best_estimator_
    rfc = final_model
    rfc.fit(X_train, Y_train)
    y_predict = rfc.predict(X_test)
    acc = accuracy_score(Y_test, y_predict)
    # fscore = f1_score(Y_test, y_predict)
    fscore = 0
    print()
    print('accuracy score', acc)
    # print('F-score', fscore)
    print()
    return [acc, fscore]


def RF_CV(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, Y_train)
    final_model = CV_rfc.best_estimator_
    rfc = final_model
    rfc.fit(X_train, Y_train)
    y_predict = rfc.predict(X_test)
    acc = accuracy_score(Y_test, y_predict)
    # fscore = f1_score(Y_test, y_predict)
    fscore = 0
    print()
    print('accuracy score', acc)
    # print('F-score', fscore)
    print()
    return [acc, fscore]
