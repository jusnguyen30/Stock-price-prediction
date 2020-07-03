import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class LinearModel:
    model = None
    x, y, x_train, x_test, y_train, y_test, y_pred = None, None, None, None, None, None, None

    def __init__(self, data):
        self.data = data
        self.model = self.linearRegression(self.data)

    def linearRegression(self, data):
        self.x = data[['prevOpen', 'prevHigh', 'prevLow', 'prevClose', 'prevAdjClose', 'prevVolume']].values
        self.y = data['AdjClose'].values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)

        regressor = LinearRegression()
        regressor.fit(self.x_train, self.y_train)

        self.y_pred = regressor.predict(self.x_test)

        df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        print(df.head())
    
        return regressor
    
    def predict(self, input):
        return self.model.predict(input)

    def getMAE(self):
        return metrics.mean_absolute_error(self.y_test, self.y_pred)

    def getMSE(self):
        return metrics.mean_squared_error(self.y_test, self.y_pred)

    def getRMSE(self):
        return np.sqrt(self.getMSE())

    def getIntercept(self):
        return self.model.intercept_

    def getCoefficients(self):
        coef_df = pd.DataFrame(self.model.coef_, ['prevOpen', 'prevHigh', 'prevLow', 'prevClose', 'prevAdjClose', 'prevVolume'], columns=['Coefficient'])
        return coef_df

    def getRSquared(self):
        return self.model.score(self.x_test, self.y_test)

    def plot(self):
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel('True values')
        plt.ylabel('Predictions')
        plt.show()