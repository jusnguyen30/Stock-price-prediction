import pandas as pd
from processData import *
from models.linearRegression import *
from models.kNearestNeighbors import *
from models.naiveBayes import *
from models.randomForest import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

googleDf = pd.read_csv("/Users/Vu/Dropbox/GOOGL.csv")
googleDf = googleDf.drop('Date', axis = 1)

# convert original time series data to supervised learning problem
googleDf = convertToSupervised(googleDf, 1, 1)

""" print(googleDf.head())
print(googleDf.describe())
print(googleDf.shape()) """

# build and train linear model 
linearModel = LinearModel(googleDf)

print("\nIntercept:", linearModel.getIntercept())
print("Coefficients:\n", linearModel.getCoefficients())
print("\nMAE:", linearModel.getMAE())
print("MSE:", linearModel.getMSE())
print("RMSE:", linearModel.getRMSE())
print("R^2 Statistic:", linearModel.getRSquared())
linearModel.plot()






""" numIn = 1
numOut = 1
numVars = googleDf.shape[1]
cols, colNames = [], []

for x in range(numIn, 0, -1):
    cols.append(googleDf.shift(x))
    colNames += [('var%d(t-%d)' % (y+1, x)) for y in range(numVars)]

for x in range(0, numOut):
	cols.append(googleDf.shift(-x))
	if x == 0:
		colNames += [('var%d(t)' % (y+1)) for y in range(numVars)]
	else:
		colNames += [('var%d(t+%d)' % (y+1, x)) for y in range(numVars)]

total = pd.concat(cols, axis = 1)
total.columns = colNames
total.dropna(inplace = True)

total = total.drop(labels = ['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var6(t)'], axis = 1)

print(total.head()) """



""" total.plot(x='var5(t-1)', y='var5(t)', style='o')
plt.show() """

