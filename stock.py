import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# getting the data
sp500 = yf.Ticker("^GSPC")
sp500 =sp500.history (period ="max") # Max period

# Show the column names
print(sp500.columns)

sp500.plot.line(y='Close', use_index=True)
plt.show()

#deleting row that arent needed
del sp500['Dividends']
del sp500['Stock Splits']

#gets the next days closing by using the shift( shifts moved everything back by one day)
sp500['Tomorrow']= sp500['Close'].shift(-1)

sp500['Target']=sp500['Tomorrow']> sp500['Close'].astype(int)

#copying/ removing everything before 1990
sp500 = sp500.loc["1990-01-01":].copy()

from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(n_estimators=100, min_samples_leaf=100, random_state=1)

train= sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors =['Close','Volume','Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

from sklearn.metrics import precision_score

preds= model.predict(test[predictors])

preds=pd.Series(preds, index=test.index)
print("Accuracy: "+str(precision_score(test['Target'],preds)))

combined = pd.concat([test['Target'],preds], axis=1)

combined.plot()

def predict(train, test, predictor, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test['Target'],preds], axis=1)