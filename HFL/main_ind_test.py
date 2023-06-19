from sklearn.utils import resample

from bp import bp

import pandas as pd

from HFL.ANN import ann
from procss import divide_data
from DT_CV import DT_CV, DT
from LR_CV import LR_CV
from shapley import shap
from itertools import combinations
from RF_CV import RF_CV
from SVM_CV import SVM_CV
from sklearn.model_selection import train_test_split
from central_model import central_model, individual_evaluation
from central_model import combine_models
from sklearn.preprocessing import StandardScaler


def weight_acc(claims, prc):
    #     lamda * claims + (1-lamda) * (claims * prc)
    lamda = 0
    result = [lamda * claim + (1 - lamda) * (claim * p) for claim, p in zip(claims, prc)]
    return result


def shapley_prc(prc):
    sprc = []
    for i in range(2, len(prc) + 1):
        g = list(combinations(prc, i))
        for com in g:
            sprc.append(round(sum(com), 2))
    return sprc


def train(column_name: str, data):
    prc = [0.2, 0.3, 0.5]
    prc_shap = prc.copy()
    prc_shap.extend(shapley_prc(prc))
    contributors = divide_data(data, prc)
    acc = []
    models = []
    for cont in contributors:
        Y = cont[column_name]
        X = cont.drop([column_name], axis=1)
        model = ann()
        model.individual_test(X, Y)
        model.train()
        models.append(model.get_model())
        # acc.append(model.get_acc()[0])
        acc.append(model.get_acc()[1])

    shaply = acc.copy()
    for i in range(2, len(contributors) + 1):
        combin = list(combinations(contributors, i))
        for comb in combin:
            data = pd.concat(comb)
            y = data[column_name]
            x = data.drop([column_name], axis=1)
            model = ann()
            model.individual_test(x, y)
            model.train()
            shaply.append(model.get_acc()[1])

    estate = shaply[-1]
    # print(estate)
    shaply = weight_acc(shaply, prc_shap)
    shaply = [round(num,4) for num in shaply]

    claims = weight_acc(acc, prc)
    sorted_list = sorted(enumerate(claims), key=lambda x: x[1])
    claims = [value for index, value in sorted_list]
    sorted_indices = [index for index, value in sorted_list]
    claims = [round(num, 4) for num in claims]
    claims = [0 if i < 0 else i for i in claims]

    print()
    print('shapley claims', shaply)
    print("shapley", shap(shaply, len(contributors)))
    print()
    print('estate', estate)
    print('claims', claims)
    print()
    print('the order of claims', sorted_indices)
    bp(estate, claims)


if __name__ == '__main__':
    bank_data = pd.read_csv("numerical_bank_data.csv")
    random_data = pd.read_csv("data/random_generated_binary_data.csv")
    diabetes_data = pd.read_csv('data/diabetes_prediction_dataset.csv')
    diabetes_data = diabetes_data[
        ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    # diabetes_data = diabetes_data[:1000]
    #

    train("Exited", bank_data)
    # train("diabetes", diabetes_data)
    # train("6", random_data)c



    # federated_model = combine_models(models)
    # estate = individual_evaluation(models, federated_model)[0]
    # cm = ann()
    # cm.individual_test(data.drop([column_name], axis=1), data[column_name])
    # cm.train()
    # estate = cm.get_acc()
    # estate = cm.get_acc()[0]