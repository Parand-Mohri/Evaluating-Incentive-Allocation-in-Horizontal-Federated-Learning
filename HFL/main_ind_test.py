import random
from itertools import combinations

import pandas as pd
from procss import divide_data
from DT_CV import DT_CV, DT
from LR_CV import LR_CV
from talmud_debts import debts
from shapley import shap
from RF_CV import RF_CV
from SVM_CV import SVM_CV


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



def chess_data():
    chess_data = pd.read_csv("data/chess_games.csv")
    chess_data = chess_data[["rated", "turns", "white_rating", "black_rating", "opening_moves", "winner"]]
    # chess_data = chess_data.loc[chess_data['winner'].isin(['Black', 'White'])]
    # chess_data = chess_data.replace('White', 0)
    # chess_data = chess_data.replace('Black', 1)
    # chess_data = chess_data[:2000]
    contributors = divide_data(chess_data, prc=[0.2,0.3,0.5])
    acc = []
    # contributors = [contributors[2],contributors[2]]
    for cont in contributors:
        Y = cont['winner']

        X = cont.drop(['winner'], axis=1)
        # acc.append(DT_CV(X, Y)[1])
        acc.append(DT_CV(X, Y)[0])
        # acc.append(RF_CV(X, Y)[0])

    Y_c = chess_data['winner']
    X_c = chess_data.drop(['winner'], axis=1)
    Estate = DT_CV(X_c, Y_c)[1]
    # Estate = RF_CV(X_c, Y_c)[0]
    claims = sorted([round(num, 2) for num in acc])
    print()
    print('estate', Estate)
    print('claims', claims)
    print()
    print(debts(Estate, claims))


def random_gen_data():
    random_gen_data = pd.read_csv("data/random_generated_data.csv")
    random_gen_data = random_gen_data[:60000]
    contributors = divide_data(random_gen_data, prc=[0.005, 0.01])
    acc = []
    for cont in contributors:
        Y = cont['6']
        X = cont.drop(['6'], axis=1)
        # acc.append(DT_CV(X, Y)[0])
        acc.append(SVM_CV(X, Y)[0])
        # acc.append(LR_CV(X, Y)[1])

    Y_c = random_gen_data['6']
    X_c = random_gen_data.drop(['6'], axis=1)
    # Estate = LR_CV(X, Y)[1]
    # Estate = DT_CV(X_c, Y_c)[0]
    Estate = SVM_CV(X_c, Y_c)[0]
    claims = sorted([round(num, 2) for num in acc])
    print()
    print('estate', Estate)
    print('claims', claims)
    print()
    print(debts(Estate, claims))


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


if __name__ == '__main__':
    # print(debts(28.03,[27.85,20.17,15.84]))
    # with chess data
    # chess_data()
    # with random generated data
    random_gen_data()
    # chess_data_shap()
    # random_gen_data_shap()
