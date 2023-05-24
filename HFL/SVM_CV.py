import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVC

def SVM_CV(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    svm = SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    final_model = grid_search.best_estimator_
    svm = final_model
    svm.fit(X_train, Y_train)
    y_predict = svm.predict(X_test)
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