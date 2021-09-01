from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# path for data
url = '/home/fitz/Documents/programming/ml/projeto/winequality-red.csv'

col_names = ['fixed acidity','volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol', 'quality']

feature_cols = ['fixed acidity','volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                'sulphates', 'alcohol']

# carregar base de dados
dataset = pd.read_csv(url, sep=";", header=None, names=col_names)

y = dataset.quality
x = dataset[feature_cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)

# Arvore de decisao usando o crirerio entropy

model_entropy = tree.DecisionTreeClassifier(criterion='entropy')
model_entropy = model_entropy.fit(x_train, y_train)
entropy_result = model_entropy.predict(x_test)
acc_entropy = metrics.accuracy_score(entropy_result, y_test)
show_entropy = f'Entropy result: {round(acc_entropy * 100)}%'

print(" === Entropy === ")
print(show_entropy)
print(list(entropy_result)[:10])
print(list(y_test)[:10])

print()

# Arvore de decisao usando o criterio do gini

#model_gini = tree.DecisionTreeClassifier() # gini é o default
model_gini = tree.DecisionTreeClassifier(criterion='gini')
model_gini = model_gini.fit(x_train, y_train)
gini_result = model_gini.predict(x_test)
acc_gini = metrics.accuracy_score(gini_result, y_test)
show_gini = f'Gini result: {round(acc_gini * 100)}%'

print(" === Gini === ")
print(show_gini)
print(list(gini_result)[:10])
print(list(y_test)[:10])

print()

k1 = 3
k2 = 5
k3 = 8

# metric1 = "euclidean" # sempre tem dados resultado igual ao do minkowski
metric1 = "chebyshev"
metric2 = "minkowski"

# Knn usando a metrica euclidiana com 3 vizinhos (k)

model_knn_euclidean1 = KNeighborsClassifier(n_neighbors=k1, metric=metric1, algorithm='brute')
model_knn_euclidean1 = model_knn_euclidean1.fit(x_train, y_train)
knn_result_euclidean1 = model_knn_euclidean1.predict(x_test)
acc_knn_euclidean1 = metrics.accuracy_score(knn_result_euclidean1, y_test)
show_knn_euclidean1 = f'Knn result: {round(acc_knn_euclidean1 * 100)}%'

print()

print(f" === Knn com {k1} vizinhos usando a metrica {metric1} === ")
print(show_knn_euclidean1)
print(list(knn_result_euclidean1)[:10])
print(list(y_test)[:10])

# knn usando a metrica de minkowski com 3 vizinhos

model_knn_minkowski1 = KNeighborsClassifier(n_neighbors=k1, metric=metric2, algorithm='brute')
model_knn_minkowski1 = model_knn_minkowski1.fit(x_train, y_train)
knn_result_minkowski1 = model_knn_minkowski1.predict(x_test)
acc_knn_minkowski1 = metrics.accuracy_score(knn_result_minkowski1, y_test)
show_knn_minkowski1 = f'Knn result: {round(acc_knn_minkowski1 * 100)}%'

print()

print(f" === Knn com {k1} vizinhos usando a metrica {metric2} === ")
print(show_knn_minkowski1)
print(list(knn_result_minkowski1)[:10])
print(list(y_test)[:10])
