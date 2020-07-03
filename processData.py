import pandas as pd

def convertToSupervised(data, numIn=1, numOut=1):
    numVars = data.shape[1]
    cols, colNames = [], []

    for x in range(numIn, 0, -1):
        cols.append(data.shift(x))
        colNames += [('var%d(t-%d)' % (y+1, x)) for y in range(numVars)]

    for x in range(0, numOut):
	    cols.append(data.shift(-x))
	    if x == 0:
		    colNames += [('var%d(t)' % (y+1)) for y in range(numVars)]
	    else:
		    colNames += [('var%d(t+%d)' % (y+1, x)) for y in range(numVars)]

    total = pd.concat(cols, axis = 1)
    total.columns = colNames
    total.dropna(inplace = True)

    total = total.drop(labels = ['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var6(t)'], axis = 1)

    colNames = ['prevOpen', 'prevHigh', 'prevLow', 'prevClose', 'prevAdjClose', 'prevVolume', 'AdjClose']
    total.columns = colNames

    return total
