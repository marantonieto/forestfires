import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Importando o df
dataframe = pd.read_csv('forestfires.csv')

# preparando os dados e fazendo a verificação cruzada:

X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

dataframe.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun',' jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), inplace=True)

dataframe.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'),
                       (1, 2, 3, 4, 5, 6, 7), inplace=True)


trainX, testX, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2)

# implementação do modelo (Regressão Multilinear)

regressor = LinearRegression()
regressor.fit(trainX, train_y)

# dados previstos pelo modelo

y_pred = regressor.predict(testX)

# Impressão dos dados reais x dados previstos pelo modelo

for i in range(len(y_pred)):
    teste_y = test_y[i]
    y_predd = y_pred[i]
    mistakes = abs((test_y[i] - y_pred[i]) / y_pred[i] * 100)
    print(f'y_test: {teste_y:.2f} y_previsto: {y_predd:.2f} miss: {mistakes:.5}')

# por meio dessa impressão, podemos reparar que o modelo erra consideravelmente, indicando algum problema nos dados
# sendo bem específico, como eu utilizei um modelo que considera que todas as variáveis são INDEPENDENTES, a baixa taxa
# de acerto da variável alvo indica uma forte dependência entre as variáveis.

# para constatar isso de fato, eu decidi utilizar uma biblioteca estatística em python chamada StatsModel

x_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(y, x_opt).fit()
print(regressor_OLS.summary())

# Pelo condicionamento numérico, é mister que as variáveis são extremamente dependentes entre si, sendo assim
# qualquer modelo linear apresentará um alto RMSE

print("\nRSME =", np.sqrt(np.mean((test_y-y_pred)**2)))
print('\nR2 =', 1 - (np.sum((test_y - y_pred)**2) / np.sum((test_y - np.mean(test_y))**2)))
