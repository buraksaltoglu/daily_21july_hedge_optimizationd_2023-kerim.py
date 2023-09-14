import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import ggplot, aes, geom_line
from plotnine import *
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# for matrix math
import numpy as np
# for normalization + probability density function computation
from scipy import stats
# for data preprocessing
import pandas as pd
from math import sqrt, log, exp, pi
from random import uniform
from sklearn.metrics import mean_squared_error
print("import done")
from mpl_toolkits import mplot3d
from datetime import datetime
from itertools import chain
from matplotlib import cm
from datetime import datetime
# Data set Reuters whole vol surface ATM D25 D75 are critical 30 60 180 days maturities
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv(r'data_macro_finance_9_1', encoding='utf-8', sep=';', parse_dates=['date'])
(data.info())


datar = data.drop(['date'], axis=1)
datam.set_index('date', inplace=True)

datam.index = pd.to_datetime(data.index)
#datam = (datam.resample('1D').mean())  # converting to month



data = data.dropna()

data.head()
hist_ret = datar['ATM60']
model = MarkovRegression(endog=dif_datam['brent'], k_regimes=2, trend='n', exog=dif_datam['CO1'],
                         switching_variance=True)
model_ols= sm.OLS(dif_datam['brent'], dif_datam['CO1'])
res_ols=model_ols.fit()
print('------in sample MSE OLS vs RS-------*************')
res = model.fit()
prediction_rs=(res.predict())
prediction_ols=(res_ols.predict())
mse_rs = sklearn.metrics.mean_squared_error(dif_datam['brent'], prediction_rs)
mse_ols = sklearn.metrics.r2_score(dif_datam['brent'], prediction_ols)

print(mse_rs)
print(mse_ols)
(prediction_rs-dif_datam['brent']).plot()
plt.show()
res.summary()
print(res.summary())


print(res.expected_durations)
print((res.smoothed_marginal_probabilities[1]))
x_list = np.arange(0, len(datar)).tolist()


(res.smoothed_marginal_probabilities[0]).plot(
    title="Probability of being in the high regime", figsize=(12, 3)
)

plt.show()