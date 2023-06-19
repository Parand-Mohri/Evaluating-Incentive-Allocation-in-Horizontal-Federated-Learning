from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,cohen_kappa_score


def LR_CV(X, Y) -> float:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)
    print('xtraine', len(X_train))
    print('xtest', len(X_test))
    param_grid = {'C': [0.001,0.01,0.1,1,10,100,1000]}
    reg = LogisticRegression()
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    final_model = grid_search.best_estimator_
    reg = final_model
    reg.fit(X_train, Y_train)
    y_predict = reg.predict(X_test)
    # acc=accuracy_score(Y_test, y_predict)
    kappa = cohen_kappa_score(Y_test,y_predict)
    fscore = f1_score(Y_test, y_predict)
    print()
    # print('accuracy score',acc )
    print('F-score',fscore )
    print()
    return [kappa, fscore]
