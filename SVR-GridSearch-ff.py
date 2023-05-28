import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Importando o df

dataframe = pd.read_csv('forestfires.csv')

# Garantindo que qualquer csv/data que for inserido não terá nenhum valor nulo a ser analisado, isso porque os valores
# nulos podem atrapalhar significativamente a análise do modelo

if dataframe.isnull().values.any():
    dataframe.interpolate()

# a escolha do interpolate deu-se pela tentativa de otimizar o processamento dos dados, isso porque assumir que um valor
# null é zero, pra mim, é significativamente prejudicial nesse tipo de questão, alterando significativamente as
# previsões do modelo

# Depois de tentar exaustivamente procurar uma relação entre (area x days e area x 2th) eu percebi que a variável alvo
# 'area' estava muito grande em comparação com as variáveis numéricas do dia e do mês, fazendo com que encontrar uma re-
# lação entre elas torna-se muito difícil, então eu decidi diminuir significativamente o tamanho de todos os valores
# da coluna area, seguindo a mesma fórmula cedida pela documentação da database (lnx + 1)

dataframe['log_area'] = np.log(dataframe['area']+1)

# verificando a relação das variáveis mutáveis com a nova variável alvo (log_area)
for i in dataframe.describe().columns[:-2]:
    dataframe.plot.scatter(i, 'log_area', grid=True, color='red')
    # plt.show()

dataframe.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun',' jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), inplace=True)

dataframe.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'),
                       (1, 2, 3, 4, 5, 6, 7), inplace=True)

# Ao analisar exaustivamente os dados, inicialmente me veio o questionamento se a variável 'dia' era realmente "útil"
# para o meu modelo, isto porque, inicialmente, eu não via sentido em utilizá-la. Logo, ao utilizar os boxplots,
# consegui atestar que elas possuem sim, alguma relação com a variável 'area' porém, pelo fato da 'area' ser muito grand
# em comparação com a variável, a visualização dessa relação era muito difícil, por isso a diminuição e a criação
# da coluna extra (log_area)

dataframe.boxplot(column='log_area', by='day')
 # plt.show()


# Preparação dos dados para a inserção no modelo de regressão

X = dataframe.drop('area', axis=1)
y = dataframe['log_area']


# StandardScaler servirá para organizar e padronizar os dados tanto de y, quanto de X quando eles forem para o modelo
sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)

# é necessário usar o método np reshape em y_train pois o método fit exige que seja inserido uma matriz e y_train é um
# vetor
y_train = y_train.values.reshape(y_train.size, 1)

# optei por utilizar o GridSearch por ser um dos modelos que, inicialmente deu um bom resultado (RMSE = 20 na primeira
# tentativa), porém ao estudar mais os parâmetros eu consegui cada vez mais melhorar o desempenho do modelo e
# utilizando cada vez mais argumentos em 'params' eu consegui, em troca de tempo de processamento, uma melhor precisão

params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 1.0, 10.0, 100.0, 1000.0],
            'epsilon': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0],
              'kernel': ['rbf', 'linear', 'poly']}

# Implementação do modelo:

grid_svr = GridSearchCV(SVR(), params, refit=True)

grid_svr.fit(sc.fit_transform(X_train), sc.fit_transform(y_train))

# o altíssimo tempo de processamento se dá pelo fato de que o gridsearch faz uma busca exaustiva pela melhor combinação
# de parâmetros, testando todas as combinações possíveis e, retornando a melhor delas. Como eu inseri em 'params'
# muitos parâmetros, o modelo vai testas todas as combinações possíveis dentre os mencionados.

y_pred = grid_svr.predict(X_test)

# melhores parâmetros encontrados pelo modelo:

best_params = grid_svr.best_params_

# quanto mais diminuirmos o epsilon, mais preciso será o modelo. Não é necessário adicionar apenas um valor menor para
# epsilon no dict 'params', para otimizar o modelo em questão de tempo de processamento, seria necessário apenas passar
# apenas os melhores parâmetros. Os deixei para mostrar como foi toda a cadeia de raciocínio até chegar nessa conclusão

print("Melhores parâmetros para esse dataset: ", best_params)

print("\nRSME =", np.sqrt(np.mean((y_test-y_pred)**2)))

print('\nR2 =', 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)))

