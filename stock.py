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

# combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test['Target'],preds], axis=1)
    return combined

def backtest(data,model, predictors, start=2500, step=250):
    all_prediction=[]

    for i in range(start, data.shape[0], step):
        train= data.iloc[0:i].copy()
        test= data.iloc[i:(i+step)].copy()
        predictions= predict(train, test, predictors,model)
        all_prediction.append(predictions)

    return pd.concat(all_prediction)

predictions = backtest(sp500, model, predictors)

predictions["Predictions"].value_counts()

#precision_score(predictions['Target'].value_counts()/predictions.shape[0])

horizons= [2,5,60,250,1000]
new_predictors= []

for horizons in horizons:
    rolling_averages= sp500.rolling(horizons).mean()

    ratio_column= f"Close_Ratio_{horizons}"
    sp500[ratio_column]=sp500['Close']/rolling_averages['Close']

    trend_column = f"Trend_{horizons}"
    sp500[trend_column]=sp500.shift(1).rolling(horizons).sum()['Target']

    new_predictors += [ratio_column, trend_column]

sp500=sp500.dropna()

model= RandomForestClassifier(n_estimators=500, min_samples_leaf=50,random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6]=1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test['Target'],preds], axis=1)
    return combined

predictions = backtest(sp500,model, new_predictors)
predictions["Predictions"].value_counts()

print(str(precision_score([predictions['Target']],predictions['Predictions'])))