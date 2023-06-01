import itertools
import random
from itertools import combinations
from bp import bp

import pandas as pd
import numpy as np

from HFL.ANN import ann
from procss import divide_data
from DT_CV import DT_CV, DT
from LR_CV import LR_CV
from shapley import shap
from RF_CV import RF_CV
from SVM_CV import SVM_CV
from sklearn.model_selection import train_test_split
from central_model import central_model
from central_model import combine_models
from sklearn.preprocessing import StandardScaler

def standarize_data(X_train, X_test):
    ss = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.columns)
    return X_train, X_test


def central_test(data, column_name):
    Y_c = data[column_name]
    X_c = data.drop([column_name], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X_c, Y_c, test_size=0.2, random_state=42)
    X_train, X_test = standarize_data(X_train, X_test)

    data = X_train.copy()
    data[column_name] = Y_train.values

    return [data, [X_test, Y_test]]


def weight_acc(claims, prc):
    #     lamda * claims + (1-lamda) * (claims * prc)
    lamda = 1
    result = [lamda * claim + (1 - lamda) * (claim * p) for claim, p in zip(claims, prc)]
    return result


def train(cen_test, column_name: str, data):
    if cen_test:
        data, test_data = central_test(data, column_name)
    prc = [0.1, 0.9]
    contributors = divide_data(data, prc)
    acc = []
    models = []
    for cont in contributors:
        Y = cont[column_name]
        X = cont.drop([column_name], axis=1)
        model = ann()
        if cen_test:
            model.central_test(X_train=X, Y_train=Y, X_test=test_data[0], Y_test=test_data[1])
        else:
            model.individual_test(X, Y)
        model.train()
        models.append(model.get_model())
        acc.append(model.get_acc()[0])

    if cen_test:
        estate = (central_model(models, test_data[0], test_data[1], prc )[0])
        # estate = combine_models(models, test_data[0], test_data[1])[1]
    else:
        # TODO: N accuracy for central model
        estate = -1

    claims = sorted([round(num, 2) for num in acc])
    claims = [0 if i < 0 else i for i in claims]
    claims = weight_acc(claims, prc)
    print()
    print('estate', estate)
    print('claims', claims)
    print()
    bp(estate, claims)


# if __name__ == '__main__':
#     chess_data = pd.read_csv("data/chess_useful_data.csv")
#
#     train(True, "winner", chess_data)

# if __name__ == '__main__':
#     random_gen_data = pd.read_csv("data/random_generated_binary_200000_noisy_data.csv")
#
#     train(cen_test=True, column_name="6", data=random_gen_data)

if __name__ == '__main__':
    diabetes_data = pd.read_csv("data/diabetes_prediction_dataset.csv")
    diabetes_data = diabetes_data[
        ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    train(True, "diabetes", diabetes_data)

def chess_data_shap():
    chess_data = pd.read_csv("data/chess_games.csv")
    chess_data = chess_data[["rated", "turns", "white_rating", "black_rating", "opening_moves", "winner"]]
    # chess_data = chess_data.loc[chess_data['winner'].isin(['Black', 'White'])]
    # chess_data = chess_data.replace('White', 0)
    # chess_data = chess_data.replace('Black', 1)
    # chess_data = chess_data[:20000]
    contributors = divide_data(chess_data, prc=[0.2, 0.8])
    d = []
    for cont in contributors:
        Y = cont['winner']
        X = cont.drop(['winner'], axis=1)
        d.append(DT_CV(X, Y)[1])
        # d.append(LR_CV(X, Y)[0])
    i = 0
    while i < len(contributors):
        j = i + 1
        while j < len(contributors):
            Y = pd.concat([contributors[i], contributors[j]])['winner']
            X = pd.concat([contributors[i], contributors[j]]).drop(['winner'], axis=1)
            d.append(DT(X, Y)[0])
            j += 1
        i += 1
    Y_c = chess_data['winner']
    X_c = chess_data.drop(['winner'], axis=1)
    d.append(DT_CV(X_c, Y_c)[0])
    d = [round(num, 2) for num in d]
    print(d)
    print(shap(d, len(contributors)))


def random_gen_data_shap():
    random_gen_data = pd.read_csv("data/random_generated_data.csv")
    random_gen_data = random_gen_data[:60000]
    contributors = divide_data(random_gen_data, prc=[0.2, 0.3, 0.5])
    d = []
    for cont in contributors:
        Y = cont['6']
        X = cont.drop(['6'], axis=1)
        d.append(DT_CV(X, Y)[0])
        # d.append(LR_CV(X, Y)[1])
    i = 0
    while i < len(contributors):
        j = i + 1
        while j < len(contributors):
            Y = pd.concat([contributors[i], contributors[j]])['6']
            X = pd.concat([contributors[i], contributors[j]]).drop(['6'], axis=1)
            d.append(DT_CV(X, Y)[0])
            j += 1
        i += 1
    Y_c = random_gen_data['6']
    X_c = random_gen_data.drop(['6'], axis=1)
    d.append(DT_CV(X_c, Y_c)[0])
    d = [round(num, 2) for num in d]
    print(d)
    print(shap(d, len(contributors)))


# def train(d, cen_test):
#     if d == 0:
#         chess_data = pd.read_csv("data/chess_useful_data.csv")
#         # chess_data = chess_data[:2000]
#         if cen_test:
#             chess_data, test_data = central_test(chess_data)
#
#         contributors = divide_data(chess_data, prc=[0.2, 0.3, 0.5])
#         acc = []
#         models = []
#         # contributors = [contributors[2],contributors[2]]
#         for cont in contributors:
#             Y = cont['winner']
#
#             X = cont.drop(['winner'], axis=1)
#             model = ann(X, Y)
#             model.train()
#             models.append(model.get_model())
#             acc.append(model.get_acc()[0])
#             # acc.append(DT_CV(X, Y)[1])
#             # acc.append(DT_CV(X, Y)[0])
#             # acc.append(aNN(X, Y)[0])
#             # acc.append(RF_CV(X, Y)[0])
#
#         Y_c = chess_data['winner']
#         X_c = chess_data.drop(['winner'], axis=1)
#         # Estate = DT_CV(X_c, Y_c)[1]
#     elif d == 1:
#         random_gen_data = pd.read_csv("data/random_generated_binary_200000_noisy_data.csv")
#         random_gen_data = random_gen_data[:100000]
#         if cen_test:
#             random_gen_data, test_data = central_test(random_gen_data, False)
#         contributors = divide_data(random_gen_data,
#                                    prc=[0.002, 0.007, 0.011, 0.012, 0.012, 0.017, 0.02, 0.021, 0.037, 0.044, 0.047,
#                                         0.063, 0.068, 0.071, 0.077, 0.15, 0.156, 0.239])
#         models = []
#         acc = []
#         for cont in contributors:
#             Y = cont['6']
#             X = cont.drop(['6'], axis=1)
#             if cen_test:
#                 # acc.append(DT_CV(X, Y)[0])
#                 # acc.append(SVM_CV(X, Y)[0])
#                 model = ann()
#                 model.central_test(X_train=X, Y_train=Y, X_test=test_data[0], Y_test=test_data[1])
#                 model.train()
#                 models.append(model.get_model())
#                 acc.append(model.get_acc()[0])
#                 # acc.append(LR_CV(X, Y)[1])
#             else:
#                 # acc.append(DT_CV(X, Y)[0])
#                 # acc.append(SVM_CV(X, Y)[0])
#                 model = ann()
#                 model.individual_test(X, Y)
#                 model.train()
#                 models.append(model.get_model())
#                 acc.append(model.get_acc()[1])
#                 Y_c = random_gen_data['6']
#                 X_c = random_gen_data.drop(['6'], axis=1)
#                 # acc.append(LR_CV(X, Y)[1])
#     elif d == 2:
#         diabetes_data = pd.read_csv("data/diabetes_prediction_dataset.csv")
#         diabetes_data = diabetes_data[
#             ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
#         diabetes_data, test_data = central_test(diabetes_data, 2)
#         # prc=[ 0.002,0.01, 0.03,0.05, 0.1, 0.8]
#         prc = [0.2, 0.3, 0.5]
#         contributors = divide_data(diabetes_data, prc=prc)
#         acc = []
#         models = []
#         for cont in contributors:
#             Y = cont['diabetes']
#             X = cont.drop(['diabetes'], axis=1)
#             model = ann()
#             model.individual_test(X, Y)
#             model.train()
#             models.append(model.get_model())
#             acc.append(model.get_acc()[0])
#         if False:
#             for i in range(2, len(contributors)):
#                 combinations = list(itertools.combinations(range(0, len(contributors)), i))
#                 i = 0
#                 while i < len(combinations):
#                     contributors_indices = combinations[i]
#                     Y = pd.concat([contributors[idx] for idx in contributors_indices])['diabetes']
#                     X = pd.concat([contributors[idx] for idx in contributors_indices]).drop(['diabetes'], axis=1)
#                     model = ann()
#                     model.individual_test(X, Y)
#                     model.train()
#                     acc.append(model.get_acc()[0])
#                     i += 1
#
#     # Estate = aNN(X_c, Y_c)[0]
#     # Estate = RF_CV(X_c, Y_c)[0]
#     if cen_test:
#         # Estate = (central_model(models, test_data[0], test_data[1], prc )[0])
#         Estate = combine_models(models, test_data[0], test_data[1])[1]
#
#     if False:
#         acc.append(Estate)
#
#         # Estate = (central_model(models, test_data[0], test_data[1], prc )[0])
#     # else:
#     #     m = ann()
#     #     m.individual_test(X_c, Y_c)
#     #     m.train()
#     #     Estate = m.get_acc()[1]
#     # Estate = aNN(X_c, Y_c)[0]
#
#     # claims = sorted([round(num, 2) for num in acc])
#     claims = [0 if i < 0 else i for i in acc]
#     claims = weight_acc(claims, prc)
#     # claims = [x * y for x, y in zip(claims, prc)]
#     # claims = [num*100 for num in claims]
#     if False:
#         print(claims)
#         print(shap(claims, len(contributors)))
#     else:
#         print()
#         print('estate', Estate)
#         print('claims', claims)
#         print()
#         bp(Estate, claims)
