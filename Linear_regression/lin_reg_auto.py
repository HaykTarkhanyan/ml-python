import argparse
import pandas as pd
import operator
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

train_features = pd.read_csv("data\\Educ.csv")
# answers are in 'lnw' column
train_labels = train_features["lnw"]
# want need taht column since we set it to specaial variable
del train_features["lnw"]


X_train, X_test, Y_train, Y_test = train_test_split(
    train_features, train_labels, test_size=0.2)


# num_of_models = 15  # num of models we will experiment on

all_models = {}
alphas = [1 / (3**i) for i in range(15)]


print(len(dir(linear_model)))

models = dir(linear_model)
# models = dir(linear_model)[:num_of_models]
# models.remove("Hinge")
# for i in models:
#     if "huber" in i or "Huber" in i:
#         models.remove(i)
models.remove("Huber")
models.remove("PassiveAggressiveClassifier")
models.remove("MultiTaskLasso")
models.remove("MultiTaskElasticNet")
models.remove("HuberRegressor")
models.remove("SGDClassifier")
models.remove("RidgeClassifier")
models.remove("Perceptron")
# print("We will work with models below: ")
# print(models)
print(len(models))
all_scores = []


def train_model(model_name, alp):
    model = getattr(linear_model, model_name)(alp)

    model.fit(X_train, Y_train)

    ans = Y_test.to_numpy()

    performance = model.score(X_test, Y_test)

    all_models.update({(model_name, "alpha: ", alp):  performance})

    all_scores.append(performance)


def train_model_2(model_name, alp1, alp2):
    model = getattr(linear_model, model_name)(alp1, alp2)

    model.fit(X_train, Y_train)

    ans = Y_test.to_numpy()

    performance = model.score(X_test, Y_test)

    all_models.update({(model_name, "alphas: ", alp1, alp2):  performance})

    all_scores.append(performance)


for i in range(len(models)):
    try:
        model = getattr(linear_model, models[i])()
    except:
        # print(model)
        pass
    if hasattr(model, "alpha"):
        print(i)
        for alpha in alphas:
            try:
                train_model(models[i], alpha)
            except:
                pass
                # print(model)

    elif hasattr(model, "alpha_1") and hasattr(model, "alpha_2"):
        print(i)
        for alpha1 in alphas:
            for alpha2 in alphas:
                try:
                    train_model_2(models[i], alpha1, alpha2)
                except:
                    pass
    print(len(all_scores))



print("#" * 30)
print(all_scores)
print(max(all_scores))


clf = svm.SVR()
clf.fit(X_train, y_Train) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)

performance = model.score(X_test, Y_test)

# maximas = max_key = max(all_models, key=lambda k: all_models[k])
# print(maximas)
# print(all_models[maximas])

# print(all_models)

