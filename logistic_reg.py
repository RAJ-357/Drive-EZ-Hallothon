
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score



def log_reg(x,input_1,input_2):
    dataset = pd.read_csv(x)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

    # print(classifier.predict(sc.transform([[30,87000]])))

    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)*100

    y_pur = classifier.predict([[input_1,input_2]])  #user input goes into predict fn
    return y_pur[0],acc
    

#log_reg('Social_Network_Ads.csv')













