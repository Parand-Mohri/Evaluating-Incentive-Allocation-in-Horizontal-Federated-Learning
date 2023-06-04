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


# def central_test(data, column_name):
#     Y_c = data[column_name]
#     X_c = data.drop([column_name], axis=1)
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X_c, Y_c, test_size=0.2, random_state=42)
#     X_train, X_test = standarize_data(X_train, X_test)
#
#     data = X_train.copy()
#     data[column_name] = Y_train.values
#
#     return [data, [X_test, Y_test]]


def weight_acc(claims, prc):
    #     lamda * claims + (1-lamda) * (claims * prc)
    lamda = 1
    result = [lamda * claim + (1 - lamda) * (claim * p) for claim, p in zip(claims, prc)]
    return result


def shapley_prc(prc):
    sprc = []
    for i in range(2, len(prc) + 1):
        g = list(combinations(prc, i))
        for com in g:
            sprc.append(round(sum(com), 2))
    return sprc


def train(cen_test, column_name: str, data):
    # if cen_test:
    #     data, test_data = central_test(data, column_name)
    prc = [0.1, 0.2, 0.3, 0.4]
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
        acc.append(model.get_acc()[0])

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
            shaply.append(model.get_acc()[0])

    # federated_model = combine_models(models)
    # estate = individual_evaluation(models, federated_model)[0]
    # cm = ann()
    # cm.individual_test(data.drop([column_name], axis=1), data[column_name])
    # cm.train()
    # estate = cm.get_acc()
    # estate = cm.get_acc()[0]
    estate = shaply[-1]
    # print(estate)
    shaply = weight_acc(shaply, prc_shap)
    # claims = sorted([round(num, 4) for num in acc])
    # claims = sorted(acc)
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
    # shapley_prc([0.1, 0.2, 0.3, 0.4], 4)
    diabetes_data = pd.read_csv("data/diabetes_prediction_dataset.csv")
    diabetes_data = diabetes_data[
        ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    # diabetes_data = diabetes_data[:1000]
    #
    train(False, "diabetes", diabetes_data)


# if __name__ == '__main__':
#     random_gen_data = pd.read_csv("numerical_bank_data.csv")
#     # test_data = pd.read_csv("data/testdata.csv")
#     # CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited,Complain,Satisfaction Score,Card Type,Point Earned
#     train(False, column_name="Exited", data=random_gen_data)


# if __name__ == '__main__':
#     # diabetes_data = pd.read_csv("data/diabetes_prediction_dataset.csv")
#     # diabetes_data = diabetes_data[
#     #     ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
#     diabetes_data = pd.read_csv('data/random_generated_binary_200000_new_data.csv')
#     # Separate the majority and minority class
#     majority_class = diabetes_data[diabetes_data['6'] == 0]
#     minority_class = diabetes_data[diabetes_data["6"] == 1]
#
#     # Undersample the majority class
#     majority_downsampled = resample(majority_class,
#                                     replace=False,  # Set to False for undersampling
#                                     n_samples=len(minority_class),
#                                     # Match the number of instances in the minority class
#                                     random_state=42)  # Set a random state for reproducibility
#
#     # Combine the downsampled majority class with the original minority class
#     balanced_df = pd.concat([majority_downsampled, minority_class])
#
#     # Shuffle the dataset
#     balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
#
#     # The balanced dataset now has an equal number of 0s and 1s
#     # print("Balanced Dataset:")
#     # print(balanced_df['diabetes'].value_counts())
#
#     train(False, "6", balanced_df)

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
