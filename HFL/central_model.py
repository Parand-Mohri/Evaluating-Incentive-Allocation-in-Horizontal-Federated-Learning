from statistics import mean

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from sklearn.neural_network import MLPClassifier


def combine_models(models, X_test, Y_test):
    # Get the number of models
    num_models = len(models)

    # Create a new model by averaging the predictions of the existing models
    new_model = MLPClassifier()  # Replace with your desired MLPClassifier configuration

    # Iterate over the models
    for model in models:
        # Add the predictions of each model to the new model
        new_model.predict_proba = lambda X: sum(model.predict_proba(X) for _ in range(num_models)) / num_models

    # return new_model
    X_dummy = np.zeros((1, 7))
    y_dummy = np.zeros(1)
    new_model.fit(X_dummy, y_dummy)
    y_predict = new_model.predict(X_test)
    acc = accuracy_score(Y_test, y_predict)
    kappa = cohen_kappa_score(Y_test, y_predict)
    fscore = f1_score(Y_test, y_predict)
    # fscore = 0
    print()
    print('accuracy score', acc)
    print("kappa", kappa)
    print('F-score', fscore)
    print()
    return [kappa, acc, fscore]


def central_model(models, X_test, Y_test, prc):
    my_list=[]
    for model in models:
        my_list.append(model.predict(X_test))
    y_predict = []
    # print(numpy.unique(my_list[2]))
    for x in range(len(my_list[0])):
        ith_row = [column[x] for column in my_list]
        indexes_0 = [index for index, item in enumerate(ith_row) if item == 0]
        indexes_1 = [index for index, item in enumerate(ith_row) if item == 1]
        total_sum_0 = 0
        total_sum_1 = 0
        if len(indexes_0) > 0:
            total_sum_0 = sum(prc[index] for index in indexes_0)
        if len(indexes_1) > 0:
            total_sum_1 = sum(prc[index] for index in indexes_1)
        if total_sum_1 > total_sum_0:
            y_predict.append(1)
        else:
            y_predict.append(0)

        # same_index = []
        # for i in range(len(my_list)):
        #     for j in range(i + 1, len(my_list)):
        #         if my_list[i][x] == my_list[j][x]:
        #             same_index.append([i,j])
        # max_sum = float('-inf')
        # max_array = None
        # if len(same_index) == 0:
        #     y_predict.append(my_list[len(my_list)-1][x])
        # else:
        #     for array in same_index:
        #         current_sum =prc[0] + prc[1]
        #         if current_sum > max_sum:
        #             max_sum = current_sum
        #             max_array = array
        #     y_predict.append(my_list[max_array[0]][x])

    # print(len(y_predict))
    acc = accuracy_score(Y_test, y_predict)
    kappa = cohen_kappa_score(Y_test, y_predict)
    # fscore = f1_score(Y_test, y_predict)
    # fscore = 0
    print()
    print('accuracy score', acc)
    print("kappa", kappa)
    # print('F-score', fscore)
    print()
    return [ acc, kappa]
    # return [0,0]


def individual_evaluation(models, federated_model):
    acc = []
    for model in models:
        x_test, y_test = model.get_test_data()
        y_predict = federated_model.predict(x_test)
        acc.append(accuracy_score(y_test, y_predict))
    return mean(acc)