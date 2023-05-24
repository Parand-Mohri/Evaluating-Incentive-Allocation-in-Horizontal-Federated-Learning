import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


def DT_cen_test(X_train, Y_train, X_test, Y_test):
    tree_clas = DecisionTreeClassifier(max_depth=8, criterion='entropy')
    tree_clas.fit(X_train, Y_train)
    y_predict = tree_clas.predict(X_test)
    acc = accuracy_score(Y_test, y_predict)
    kappa = cohen_kappa_score(Y_test, y_predict)
    # fscore = f1_score(Y_test, y_predict)
    fscore = 0
    print()
    print('accuracy score', acc)
    # print('F-score', fscore)
    print("kappa", kappa)
    print()
    return [kappa, acc]


def DT(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    tree_clas = DecisionTreeClassifier(max_depth=8, criterion='entropy')
    tree_clas.fit(X_train, Y_train)
    y_predict = tree_clas.predict(X_test)
    acc = accuracy_score(Y_test, y_predict)
    # fscore = f1_score(Y_test, y_predict)
    fscore = 0
    print()
    print('accuracy score', acc)
    # print('F-score', fscore)
    print()
    return [acc, fscore]


def DT_CV(X, Y) -> float:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    param_grid = {'max_features': ['sqrt', 'log2'],
                  'ccp_alpha': [0.1, .01, .001],
                  'max_depth': [5, 6, 7, 8, 9],
                  'criterion': ['gini', 'entropy']
                  }
    tree_clas = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    final_model = grid_search.best_estimator_
    tree_clas = final_model
    tree_clas.fit(X_train, Y_train)
    y_predict = tree_clas.predict(X_test)
    acc = accuracy_score(Y_test, y_predict)
    kappa = cohen_kappa_score(Y_test, y_predict)
    # fscore = f1_score(Y_test, y_predict)
    fscore = 0
    print()
    print('accuracy score', acc)
    print("kappa", kappa)
    # print('F-score', fscore)
    print()
    return [kappa, acc]
