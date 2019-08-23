import os
import argparse
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# folder where data is
LOAD_DATA_FROM = 'data'

train_features = pd.read_csv(os.path.join(LOAD_DATA_FROM, "Educ.csv"))
# answers are in 'lnw' column
train_labels = train_features["lnw"]
# want need taht column since we set it to specaial variable
del train_features["lnw"]


X_train, X_test, Y_train, Y_test = train_test_split(
    train_features, train_labels, test_size=0.2)

parser = argparse.ArgumentParser()

parser.add_argument('-input', type=str, required=True,
                    help="give a list of inputs")

args = parser.parse_args()


to_test = eval(args.input)

if len(to_test) != 4:
    print("Wrong input, try something like [1,2,4,3]")
else:
    print("Using model -  Lasso, with alpha - 0.001")

    model_1 = linear_model.Lasso(alpha=0.001)

    model_1.fit(X_train, Y_train)
    print("Score is - " + str(model_1.score(X_test, Y_test)))
    print()
    to_test = np.array(to_test).reshape(1, -1)
    print(model_1.predict(to_test))

# Results of experiments.
# alpha params - 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003
# best result by model
# Lasso - 0.12
# Ridge - 0.117
# Bayesian Ridge - 0.118
# Elastik Net - 0.115
# Huber Regressior - 0.107
# SGDRegressor - 0.03
