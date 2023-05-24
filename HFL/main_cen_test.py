import random
from itertools import combinations

import pandas as pd
from sklearn.model_selection import train_test_split

from procss import divide_data
from DT_CV import DT_CV, DT, DT_cen_test
from LR_CV import LR_CV
from talmud_debts import debts
from shapley import shap
from RF_CV import RF_CV_cen_test


def chess_data():
    chess_data = pd.read_csv("data/chess_games.csv")
    chess_data = chess_data[["rated", "turns", "white_rating", "black_rating", "opening_moves", "winner"]]
    # chess_data = chess_data.loc[chess_data['winner'].isin(['Black', 'White'])]
    # chess_data = chess_data.replace('White', 0)
    # chess_data = chess_data.replace('Black', 1)
    # chess_data = chess_data[:4000]
    Y_c = chess_data['winner']
    X_c = chess_data.drop(['winner'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X_c, Y_c, test_size=0.2, random_state=42)
    data = X_train.copy()
    data['winner'] = Y_train
    contributors = divide_data(data, prc= [0.005, 0.01])

    acc = []
    # contributors = [contributors[2],contributors[2]]
    for cont in contributors:
        Y = cont['winner']

        X = cont.drop(['winner'], axis=1)
        # acc.append(DT_CV(X, Y)[1])
        acc.append(RF_CV_cen_test(X, Y, X_test, Y_test)[0])

    Estate = RF_CV_cen_test(X_train, Y_train, X_test, Y_test)[0]
    # Estate = DT_cen_test(X_c, Y_c, X_test, Y_test)[0]
    claims = sorted([round(num, 2) for num in acc])
    print()
    print('estate', Estate)
    print('claims', claims)
    print()
    print(debts(Estate, claims))


if __name__ == '__main__':
    # print(debts(28.03,[27.85,20.17,15.84]))
    # with chess data
    chess_data()
    # with random generated data
    # random_gen_data()
    # chess_data_shap()
