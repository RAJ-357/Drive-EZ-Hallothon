import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def naive_bayes(csv,input_1,input_2):
    dataset = pd.read_csv(csv)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # print("enter two independent variables")
    from sklearn.metrics import accuracy_score
    y_pred = classifier.predict(X_test)
    # pred = predict_NB(para_1,para_2)
    return classifier.predict(sc.transform([[input_1,input_2]]))[0] , accuracy_score(y_test, y_pred) 