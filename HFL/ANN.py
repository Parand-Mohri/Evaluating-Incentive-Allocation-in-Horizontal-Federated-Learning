from sklearn import preprocessing
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score


class ann():
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42, max_iter=1000)

    def individual_test(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def central_test(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def get_model(self):
        return self.clf

    def get_test_data(self):
        return self.X_test, self.Y_test

    def train(self):
        self.clf.fit(self.X_train, self.Y_train)
        # return self.clf

    def get_acc(self):
        y_predict = self.clf.predict(self.X_test)
        acc = accuracy_score(self.Y_test, y_predict)
        kappa = cohen_kappa_score(self.Y_test, y_predict)
        fscore = f1_score(self.Y_test, y_predict)
        precision = precision_score(self.Y_test, y_predict)
        recall = recall_score(self.Y_test, y_predict)
        print()
        print('accuracy score', acc)
        print("kappa", kappa)
        print('F-score', fscore)
        print("recall", recall)
        print('precision', precision)
        print()
        return [acc, kappa, fscore]
