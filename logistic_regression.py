import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import exp as mathexp
import pandas as pd
import pickle

from utils_training import *


class LogisticRegression:

    def __init__(self, nb_iter=10000, alpha=0.05, opti_func='GD', batch_size=256, verbose=False):
        self.nb_iter = nb_iter
        self.alpha = alpha
        self.classifiers = []

        self.opti_func = opti_func
        self.batch_size = batch_size
        self.verbose = verbose

    def _sigmoid_predict(self, z):
        return 1 / (1 + mathexp(-z))

    def _cost_func(self, h, y):
        return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

    def _accuracy_score(self, X, y, train=True):
        z = np.dot(X, self.thetas)
        h = self._sigmoid(z)
        predictions = np.zeros(len(y))
        predictions[h >= 0.5] = 1
        str_result = str(np.mean(predictions == y) * 100) + " %"
        if train:
            print("\tTraining Accuracy =", str_result)
        else:
            print("\tValidation Accuracy =", str_result)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _gradientDescent(self, X, y):
        for iter in range(self.nb_iter):
            z = np.dot(X, self.thetas)
            h = self._sigmoid(z)
            gradient = (np.dot(X.T, h - y)) / y.size
            self.thetas -= self.alpha * gradient

            if ((iter - 1) % 100 == 0) and self.verbose:
                print(f'\t{iter}/{self.nb_iter} loss: {self._cost_func(h, y)} \t')
        return self.thetas

    def _miniBatchGradientDescent(self, X, y):
        len_X = len(X)
        for iter in range(self.nb_iter):
            indices = np.random.permutation(len_X)
            data, data_class = X[indices], y[indices]
            batch_size = self.batch_size
            for i in np.arange(0, X.shape[0], batch_size):
                (batch_X, batch_Y) = data[i:i + batch_size], data_class[i:i + batch_size]
                z = np.dot(batch_X, self.thetas)
                h = self._sigmoid(z)
                gradient = (np.dot(batch_X.T, h - batch_Y)) / batch_Y.size
                self.thetas -= self.alpha * gradient

            if ((iter - 1) % 100 == 0) and self.verbose:
                z = np.dot(data, self.thetas)
                h = self._sigmoid(z)
                print(f'\t{iter}/{self.nb_iter} loss: {self._cost_func(h, data_class)} \t')
        return self.thetas

    def _one_vs_all_train(self, X_train, X_validation, y_train_all, y_validation_all):
        print("One vs All :")
        prediction_train = self.one_vs_all(X_train.values)
        prediction_validation = self.one_vs_all(X_validation.values)

        y_train_all = np.argmax(y_train_all.values, axis=1)
        y_validation_all = np.argmax(y_validation_all.values, axis=1)

        print("Training Accuracy =", str(np.mean(prediction_train == y_train_all) * 100) + " %")
        print("Validation Accuracy =", str(np.mean(prediction_validation == y_validation_all) * 100) + " %")

        filename = 'finalized_model.sav'
        pickle.dump(self, open(filename, 'wb'))

    def one_vs_all(self, X_to_predict):
        result = []

        for row in X_to_predict:
            best_class = 0
            class_scores = [self._sigmoid_predict(np.dot(classifier.T, row)) for classifier in self.classifiers]
            best_class = class_scores.index(max(class_scores))
            result.append(best_class)
        return result

    def train(self, df, y):

        X_train, X_validation, y_train_all, y_validation_all = train_test_split(df, y)
        print("\nTraining in process: ")
        for i in range(-4, 0):
            print("Class: ", y_train_all.columns[i])
            y_train = y_train_all[y_train_all.columns[i]]
            y_validation = y_validation_all[y_validation_all.columns[i]]
            self.thetas = np.zeros(X_train.shape[1])

            opti_function = self.optimisation_functions[self.opti_func]
            opti_function(self, X_train.values, y_train.values)

            self._accuracy_score(X_train, y_train)
            self._accuracy_score(X_validation, y_validation, False)

            self.classifiers.append(self.thetas)

        print("Iterations per class:", self.nb_iter)
        print("Learning rate:", self.alpha)
        print("DONE\n")

        self._one_vs_all_train(X_train, X_validation, y_train_all, y_validation_all)

    optimisation_functions = {
        'GD': _gradientDescent,
        'MBGD': _miniBatchGradientDescent
        }
