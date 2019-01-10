import pandas as pd
import numpy as np


def prepare_data(data):
    data.drop(['Index', 'First Name', 'Last Name',
               'Birthday'], axis=1, inplace=True)

    dummies_hand = pd.get_dummies(data['Best Hand'], drop_first=True)

    df = pd.concat([data, dummies_hand], axis=1)
    y = pd.get_dummies(data['Hogwarts House'])

    df.drop(['Best Hand', 'Hogwarts House'], axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df = normalize(df)
    df = add_intercept(df)

    return df, y


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def add_intercept(X):
    intercept = pd.Series(data=np.ones(len(X.index)), index=X.index)
    X.insert(loc=0, column='Intercept', value=intercept)
    return X


def train_test_split(df, y):

    X_train = df.sample(frac=0.75, random_state=300)
    X_validation = df.drop(X_train.index)
    X_train.index = range(len(X_train))
    X_validation.index = range(len(X_validation))

    y_train_all = y.sample(frac=0.75, random_state=300)
    y_validation_all = y.drop(y_train_all.index)
    y_train_all.index = range(len(y_train_all))
    y_validation_all.index = range(len(y_validation_all))

    return X_train, X_validation, y_train_all, y_validation_all


def convert_result(result):
    conversion = []
    hash_house = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
    for x in result:
        conversion.append(hash_house[x])
    return conversion
