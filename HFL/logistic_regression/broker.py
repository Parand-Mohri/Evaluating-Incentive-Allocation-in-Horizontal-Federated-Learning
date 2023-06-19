from itertools import combinations
import random

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from HFL.bp import bp
from HFL.logistic_regression.ML import LogisticRegressionFederated
from HFL.logistic_regression.client import participant
from HFL.logistic_regression.server import server
from HFL.procss import divide_data, divide_data_with_gender
from HFL.shapley import shap


def weight_acc(claims, prc):
    #     lamda * claims + (1-lamda) * (claims * prc)
    lamda = 0.75
    result = [lamda * claim + (1 - lamda) * (claim * p) for claim, p in zip(claims, prc)]
    return result


def standarize_data(X_train, X_test):
    ss = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.columns)
    return X_train, X_test


def shapley_prc(prc):
    sprc = []
    for i in range(2, len(prc) + 1):
        g = list(combinations(prc, i))
        for com in g:
            sprc.append(round(sum(com), 2))
    return sprc

bank_data = pd.read_csv("../data/random_generated_binary_data.csv")
# bank_data = pd.read_csv("../numerical_bank_data.csv")
# bank_data = pd.read_csv("../data/diabetes_prediction_dataset.csv")
# bank_data = bank_data[
#         ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
# bank_data = bank_data[:30000]
prc = [0.2, 0.3, 0.5]
# prc_shap = prc.copy()
prc_shap = [0.1,0.2,0.3,0.4]
# prc_shap.extend(shapley_prc(prc))
prc_shap.extend(shapley_prc([0.1,0.2,0.3,0.4]))
# print(prc_shap)
contributor = divide_data(bank_data, prc)
# print(len(contributor[0]),len(contributor[1]),len(contributor[2]))
# contributors = divide_data_with_gender(bank_data, prc)
# for sharin data contributor 4 has 20% of contributor 3 data points
# contributor4 = contributor[2].sample(frac=0.2, random_state=42)
# print(len(contributor4), len(contributor[2]), len(contributor[0]), len(contributor[1]))
c5divide = divide_data(contributor[2], [0.2,0.8])
# contributors = [contributor4, contributor[0] , contributor[1], contributor[2]]
contributors = [c5divide[0], contributor[0] , contributor[1], c5divide[1]]
# total_length = sum(len(dataset) for dataset in contributors)
# prc = [(len(dataset) / total_length) for dataset in contributors]
# prc_shap = prc.copy()
# prc_shap.extend(shapley_prc(prc))
# print(len(contributors[0]), len(contributors[1]), len(contributors[2]), len(contributors[3]), len(bank_data))


x_train, x_test, y_train, y_test = [], [], [], []
for cont in contributors:
    X = cont.drop(['6'], axis=1)
    Y = cont['6']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test = standarize_data(X_train, X_test)
    x_train.append(X_train)
    x_test.append(X_test)
    y_train.append(Y_train)
    y_test.append(Y_test)

client1_localdata = [x_train[0], y_train[0]]
# client2_localdata = [x_train[1], y_train[1]]
# for the bad data switch 0 and 1 in client 2
# y_train[1] = y_train[1].apply(lambda x: 1 if x == 0 else 0)
# for random experiments
# y_train[1] = y_train[1].apply(lambda x: random.randint(0, 1))
client2_localdata = [x_train[1], y_train[1]]
client3_localdata = [x_train[2], y_train[2]]
client4_localdata = [x_train[3], y_train[3]]
# for the bad data switch 0 and 1 in client 2
# y_test[1]= y_test[1].apply(lambda x: 1 if x == 0 else 0)
# for random experiments
# y_test[1] = y_test[1].apply(lambda x: random.randint(0, 1))
c1 =[x_train[0], y_train[0],x_test[0],y_test[0]]
c2= [x_train[1], y_train[1],x_test[1],y_test[1]]
c3= [x_train[2], y_train[2],x_test[2],y_test[2]]
c4= [x_train[3], y_train[3],x_test[3],y_test[3]]
# cld = [c1,c2,c3]
cld = [c1,c2,c3,c4]

communication_rounds = 100
M, N = client1_localdata[0].shape


# def predict(X, theta):
#     z = np.dot(X, theta[0]) + theta[1]
#     y_pred = expit(z)
#     return y_pred


# def predict_ann(X, theta):
#     weights = theta[0]
#     bias = theta[1]
#     return np.dot(X, weights) + bias


def predict_lr(X, theta, threshold=0.5):
    z = np.dot(X, theta[0]) + theta[1]
    y_pred = expit(z)
    y_pred_binary = (y_pred >= threshold).astype(int)
    return y_pred_binary


def evaluate_federated_model():
    # theta = pd.read_csv("yes_i_did_it.csv")
    #     w = [-0.000591, -0.002401, -0.004570, 0.012209, -0.001704, 0.005964, -0.003006, -0.000231, -0.008305, -0.000832, 0.039147, -0.001434, -0.000515, 0.002141, 0.002443, -0.001692, -0.002813, -0.004332, 0.006229, -0.001355]
    #     b = -0.02825997
    # bank data
    # w = [-0.004790, -0.004243, -0.034196, 0.094221, -0.002347, 0.031041, -0.018355, -0.000566, -0.052882, 0.003801,
    #      0.349357, -0.005202, -0.001342, 0.006623, -0.000970, -0.003758, -0.001950, -0.025812, 0.044676, -0.014822]
    # b = -0.26131974
    # random data
    # w = [ 0.007171, 0.173971, -0.031771, 0.170401, 0.001035, 0.043480, 0.014606]
    # b = -0.00190154
    # bank data --> opposite data
    # w=[-0.001076, -0.001584, -0.006820, 0.033202, 0.000364, 0.014761, -0.014707, 0.000486, -0.025267, 0.003115, 0.115846, 0.001520, -0.001129, -0.002242, 0.004046, -0.000918, -0.000987, -0.010881, 0.014491, -0.001491]
    # b= -0.08574907
    # random data -->opposite data
    # w =[0.013637, 0.057199, -0.008532, 0.058226, 0.013289, 0.011384, 0.009107]
    # b = -0.00332163
    # random data -->random label
    # w = [0.006449, 0.119569, -0.020944, 0.118448, 0.004792, 0.028229, 0.012409]
    # b=-0.00875243
    # bank data -->random label
#     w = [
# 0.003953, -0.001999, -0.024227, 0.062614, 0.000076, 0.021860, -0.014309, -0.001697, -0.039332, -0.000613, 0.232164, -0.001523, -0.002249, 0.000750, 0.002072, -0.002303, -0.000598, -0.018625, 0.031287, -0.009573]
#     b = -0.17136989
#     random data ->redistributed data
    w = [0.006275, 0.176035, -0.036585, 0.171514, -0.002392, 0.043288, 0.015565]
    b =-0.01518958
#     bank data ->redistributed
#     w=[-0.003296, -0.000957, -0.038852, 0.092085, -0.005165, 0.024578, -0.023887, -0.000953, -0.052366, 0.000725, 0.353327, -0.008910, 0.006241, 0.007212, 0.001104, -0.004283, -0.004134, -0.019898, 0.042890, -0.020740]
#     b=-0.256077
#     diabetes -> opposite data
#     w =[0.002143, 0.018455, 0.015611, 0.013058, 0.016222, 0.032899, 0.033137]
#     b =-0.12247324
#     diabtes data->random label
#     w = [0.003634, 0.036143, 0.028365, 0.024853, 0.031417, 0.066053, 0.067106]
#     b=-0.24447245
#     diabetes -> data redistribution
#     w =[0.007406, 0.056909, 0.039755, 0.035816, 0.048859, 0.098812, 0.096500]
#     b=-0.36705921
#     diabetes data
#     w = [0.008002, 0.058235, 0.042446, 0.036852, 0.048832, 0.100212, 0.098180]
#     b = -0.36610427
#     bank data -> shared data points
#     w = [0.000105, -0.002014, -0.038586, 0.090189, -0.007969, 0.030062, -0.014139, -0.001232, -0.049510, 0.004556, 0.349948, -0.002285, -0.002484, 0.009986, -0.004416, -0.005208, -0.000660, -0.025362, 0.042929, -0.013698]
#     b = -0.2616082
#     random data -> shared data points
#     w = [0.018136, 0.177397, -0.032930, 0.172369, -0.001550, 0.048452, 0.008930]
    # b = -0.00143203
    # diabetes data -> shared data points
    # w= [ 0.008229, 0.056734, 0.042184, 0.039318, 0.049529, 0.097916, 0.096388]
    # b = -0.36705334
    # diaetes data -> gender
    # w =[0.003517, 0.059006, 0.042494, 0.038871, 0.051617, 0.099752, 0.100160]
    # b = -0.36460644
    # bank data -> gender
    # w=[-0.001495, -0.012102, -0.013120, 0.091925, -0.004139, 0.028758, -0.013976, -0.004973, -0.049003, 0.002359, 0.349181, -0.003911, -0.004982, 0.005395, -0.006289, 0.003514, -0.002523, -0.030786, 0.047319, -0.011662]
    # b=-0.26010408
    new_way_k = []
    new_way_f = []
    for i in range(len(x_test)):
        new_way_k.append(cohen_kappa_score(predict_lr(x_test[i], [w,b]),y_test[i]))
        new_way_f.append(f1_score(predict_lr(x_test[i], [w,b]),y_test[i]))
    # x_t = pd.concat([t for t in x_test])
    # y_pred = predict_lr(x_t, [w, b])
    # y_t = pd.concat([p for p in y_test])
    # kappa = cohen_kappa_score(y_pred, y_t)
    # fscore = f1_score(y_pred, y_t)
    # print("kappa", kappa)
    # print("fscore", fscore)
    return new_way_k, new_way_f

def make_federated_model():
    # Create client instances
    client1 = participant(LogisticRegressionFederated, data=client1_localdata)
    client2 = participant(LogisticRegressionFederated, data=client2_localdata)
    client3 = participant(LogisticRegressionFederated, data=client3_localdata)
    # client4 = participant(LogisticRegressionFederated, data=client4_localdata)

    initial_global_model = (np.zeros((N, 1)), 0)
    # Create server instance
    S = server(initial_global_model)

    for round in range(communication_rounds):
        theta = S.send_to_clients()  # In the first iteration the initial_global_model is send to all the clients from the server

        client1.receive_from_server(theta)
        client2.receive_from_server(theta)  # Clients receive global model from server and update local model
        client3.receive_from_server(theta)  # Typically this should happen parallely but it is sequential here
        # client4.receive_from_server(theta)

        client1.train()
        client2.train()  # Clients parallely train local models using local data
        client3.train()
        # client4.train()

        if round % 5 ==0:
            progress = (round + 1) / communication_rounds * 100
            print(f"Progress: {progress:.2f}%")
            # print('client 1',client1.theta[0])    # Check local model at any iteration
            # print('client 2',client2.theta[0])
            # print('client 3',client3.theta[0])
        #
        S.receive_from_clients(client1.send_to_server(), client2.send_to_server(),
                               client3.send_to_server())  # Server receives updated local models for aggregation

    print("im done")
    pd.DataFrame(S.send_to_clients()).to_csv("logistic_regression_diabetes_data_with_data_redistribution.csv")


def logist(X_train, Y_train):
    reg = LogisticRegression()
    reg.fit(X_train, Y_train)
    return reg

if __name__ == '__main__':
    # make_federated_model()
    claims_kappa = []
    claims_fscore = []
    for i in range(len(x_train)):
        model = logist(x_train[i],y_train[i])
        pred = model.predict(x_test[i])
        claims_kappa.append(cohen_kappa_score(pred,y_test[i]))
        claims_fscore.append(f1_score(pred,y_test[i]))
    shapley_kappa = claims_kappa.copy()
    shapley_fscore = claims_fscore.copy()
    for i in range(2, len(contributors)):
        combin = list(combinations(cld, i))
        for comb in combin:
            x =pd.concat([sublist[0] for sublist in comb])
            y = pd.concat([sublist[1] for sublist in comb])
            mod = logist(x,y)
            p = mod.predict(pd.concat([sublist[2] for sublist in comb]))
            shapley_kappa.append(cohen_kappa_score(p,pd.concat([sublist[3] for sublist in comb])))
            shapley_fscore.append(f1_score(p,pd.concat([sublist[3] for sublist in comb])))

    estate_kappa, estate_fscore = evaluate_federated_model()

    new_claims_fscore = estate_fscore
    new_claims_kappa = estate_kappa

    shapley_kappa.append(np.mean(estate_kappa))
    shapley_fscore.append(np.mean(estate_fscore))

    shapley_kappa = weight_acc(shapley_kappa, prc_shap)
    shapley_fscore = weight_acc(shapley_fscore, prc_shap)
    prc = [0.1,0.2,0.3,0.4]
    claims_kappa = weight_acc(claims_kappa, prc)
    claims_fscore = weight_acc(claims_fscore, prc)

    new_claims_kappa = weight_acc(new_claims_kappa,prc)
    new_claims_fscore = weight_acc(new_claims_fscore, prc)

    sorted_list_k = sorted(enumerate(claims_kappa), key=lambda x: x[1])
    claims_kappa = [value for index, value in sorted_list_k]
    claims_kappa = [0 if num < 0 else num for num in claims_kappa]
    sorted_indices_k = [index for index, value in sorted_list_k]

    sorted_list_f = sorted(enumerate(claims_fscore), key=lambda x: x[1])
    claims_fscore = [value for index, value in sorted_list_f]
    claims_fscore =[0 if num < 0 else num for num in claims_fscore]
    sorted_indices_f = [index for index, value in sorted_list_f]

    new_sorted_list_k = sorted(enumerate(new_claims_kappa), key=lambda x: x[1])
    new_claims_kappa = [value for index, value in new_sorted_list_k]
    new_claims_kappa = [0 if num < 0 else num for num in new_claims_kappa]
    new_sorted_indices_k = [index for index, value in new_sorted_list_k]

    new_sorted_list_f = sorted(enumerate(new_claims_fscore), key=lambda x: x[1])
    new_claims_fscore = [value for index, value in new_sorted_list_f]
    new_claims_fscore = [0 if num < 0 else num for num in new_claims_fscore]
    new_sorted_indices_f = [index for index, value in new_sorted_list_f]



    print("kappa score:")
    print('estate', np.mean(estate_kappa))
    print('claims 1', claims_kappa)
    print("payoff 1:")
    bp(np.mean(estate_kappa), claims_kappa)
    print('the order of claims 1', sorted_indices_k)
    print('claims 2', new_claims_kappa)
    print('payoff 2:')
    bp(np.mean(estate_kappa), new_claims_kappa)
    print('the order of claims 2', new_sorted_indices_k)
    print('shapley value', shap(shapley_kappa,len(contributors)))
    print('shapley claims', shapley_kappa)

    print()
    print("F-score:")
    print('estate', np.mean(estate_fscore))
    print('claims 1 ', claims_fscore)
    print("payoff:")
    bp(np.mean(estate_fscore), claims_fscore)
    print('the order of claims 1', sorted_indices_f)
    print('claims 2', new_claims_fscore)
    print('payoff 2:')
    bp(np.mean(estate_fscore), new_claims_fscore)
    print('the order of claims 2', new_sorted_indices_f)
    print('shapley value', shap(shapley_fscore,len(contributors)))
    print('shapley claims', shapley_fscore)



