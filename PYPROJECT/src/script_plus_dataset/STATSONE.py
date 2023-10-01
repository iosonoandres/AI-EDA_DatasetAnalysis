# Librerie necessarie:
import pandas as pd
import numpy as np

# Librerie di visualizzazione dati:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, accuracy_score

# Librerie di machine learning

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
import sklearn.svm as svm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# iniziamo con la lettura del nostro dataset
dataframe = pd.read_csv('car.csv')



print(dataframe.shape)
print(dataframe.head())
print(dataframe.isnull().sum())



# tutto ottimo, procediamo con il controllo dei valori fuori soglia

min_limit = 0  # Valore minimo accettabile generale, potremmo farne uno specifico ma teniamoci indicativi
max_limit = 5  # Valore massimo accettabile per un dataset sulle auto

# Seleziona solo le colonne numeriche
numeric_columns = ['No_of_Doors']

# Itera sulle colonne numeriche e controlla i valori fuori soglia
for column in numeric_columns:
    if column == 'No_of_Doors':
        outliers = dataframe[~dataframe[column].str.contains('5more')]
    else:
        outliers = dataframe[(dataframe[column] < min_limit) | (dataframe[column] > max_limit)]
        if not outliers.empty:
            print(f"Variabile: {column}")
            print(outliers)
            print("\n")

# vediamo come possiamo gestire le features categoriche
for column in ['Person_Capacity']:
    print(dataframe[column].unique())


fig = go.Figure()

for category in dataframe['Car_Acceptability'].unique():
    fig.add_trace(go.Violin(x=dataframe['Car_Acceptability'][dataframe['Car_Acceptability'] == category],
                            y=dataframe['Buying_Price'][dataframe['Car_Acceptability'] == category],
                            name=category, box_visible=True, meanline_visible=True))

fig.update_layout(title='Distribuzione di Buying_Price per Car_Acceptability',
                  xaxis=dict(title='Car_Acceptability'),
                  yaxis=dict(title='Buying_Price'))
fig.show()



dataframe['Car_Acceptability'].replace({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}, inplace=True)


# Esempio di analisi delle variabili categoriche tramite countplot
categorical_columns = ['Buying_Price', 'Maintenance_Price', 'No_of_Doors', 'Person_Capacity', 'Size_of_Luggage', 'Safety', 'Car_Acceptability']

plt.figure(figsize=(10, 8))
sns.violinplot(x='Car_Acceptability', y='Buying_Price', data=dataframe)
plt.title('Distribuzione di Buying_Price per Car_Acceptability')
plt.show()
# Esempio di analisi delle variabili numeriche tramite boxplot
numeric_columns = ['Buying_Price', 'Maintenance_Price']

grid = sns.FacetGrid(dataframe, col='Car_Acceptability', height=4)
grid.map(sns.histplot, 'Buying_Price')
plt.show()




for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=column, y='Car_Acceptability', data=dataframe)
    plt.title(f'Distribuzione della variabile {column} per la variabile di target')
    plt.show()

# Esempio di analisi delle variabili numeriche tramite histogram
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(x=column, data=dataframe, hue='Car_Acceptability', kde=True)
    plt.title(f'Distribuzione della variabile {column} per la variabile di target')
    plt.show()

#PRIMA DI QUESTO BISOGNA FARE EDA
dataframe['Buying_Price'].replace({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}, inplace=True)
dataframe['Maintenance_Price'].replace({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}, inplace=True)
dataframe['No_of_Doors'].replace({'5more': 5}, inplace=True)
dataframe['Person_Capacity'].replace({'more': 5}, inplace=True)
dataframe['Size_of_Luggage'].replace({'small': 0, 'med': 1, 'big': 2}, inplace=True)
dataframe['Safety'].replace({'low': 0, 'med': 1, 'high': 2}, inplace=True)

# Esempio di analisi delle correlazioni tra le variabili numeriche tramite heatmap
correlation_matrix = dataframe.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice di correlazione')
plt.show()








# tutto andato a buon fine, ma prima di procedere con lo splitting
print(dataframe.head())

X = dataframe.drop(columns= 'Car_Acceptability')
y = dataframe['Car_Acceptability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#modello regressione logistica
modelloLogistica = LogisticRegression(solver='lbfgs', max_iter=1000)
modelloLogistica.fit(X_train, y_train)
# Definisci il modello SVC
model = SVC()
model.fit(X_train,y_train)

# Effettua predizioni sui dati di test
y_pred_logistic = modelloLogistica.predict(X_test)
#Adesso predizioni Svc
y_pred_svc = model.predict(X_test)
# Valuta le prestazioni del modello
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Accuracy del modello di regressione logistica:", accuracy_logistic)
classification_report_logistic = classification_report(y_test, y_pred_logistic)
print("Report di classificazione del modello di regressione logistica:")
print(classification_report_logistic)
#ADESSO PARTE SVC
accuracy_SVC= accuracy_score(y_test,y_pred_svc)
print("Accuracy del modello SVC:", accuracy_SVC)
classification_report_SVC = classification_report(y_test, y_pred_svc)
print("Report di classificazione del modello SVC:")



#Addestramento del modello SVM con kernel rbf
param_grid_rbf = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto']
}
grid_search_rbf = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid_rbf, cv=5)
grid_search_rbf.fit(X_train, y_train)
best_model_rbf = grid_search_rbf.best_estimator_
accuracy_rbf = best_model_rbf.score(X_test, y_test)

# Addestramento del modello SVM con kernel polinomiale
param_grid_poly = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
grid_search_poly = GridSearchCV(estimator=SVC(kernel='poly'), param_grid=param_grid_poly, cv=5)
grid_search_poly.fit(X_train, y_train)
best_model_poly = grid_search_poly.best_estimator_
accuracy_poly = best_model_poly.score(X_test, y_test)

# Addestramento del modello SVM con kernel lineare
param_grid_linear = {
    'C': [0.1, 1.0, 10.0]
}
grid_search_linear = GridSearchCV(estimator=SVC(kernel='linear'), param_grid=param_grid_linear, cv=5)
grid_search_linear.fit(X_train, y_train)
best_model_linear = grid_search_linear.best_estimator_
accuracy_linear = best_model_linear.score(X_test, y_test)


# Calcolo dell'accuratezza dei modelli
accuracy_rbf = best_model_rbf.score(X_test, y_test)
accuracy_poly = best_model_poly.score(X_test, y_test)
accuracy_linear = best_model_linear.score(X_test, y_test)

# Etichette per i modelli
labels = ['SVM (kernel rbf)', 'SVM (kernel polinomiale)', 'SVM (kernel lineare)']

# Accuratezze dei modelli
accuracies = [accuracy_rbf, accuracy_poly, accuracy_linear]

# Tracciamento del grafico a barre
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies)
plt.xlabel('Modello SVM')
plt.ylabel('Accuratezza')
plt.title('Confronto delle Accuratezze tra i Modelli SVM')
plt.show()

# Calcolo delle accuratezze dei modelli sul training set
accuracy_rbf_train = best_model_rbf.score(X_train, y_train)
accuracy_poly_train = best_model_poly.score(X_train, y_train)
accuracy_linear_train = best_model_linear.score(X_train, y_train)

# Creazione dei vettori di accuratezza
accuracies_train = [accuracy_rbf_train, accuracy_poly_train, accuracy_linear_train]


# Etichette per i modelli
labels = ['SVM (kernel rbf)', 'SVM (kernel polinomiale)', 'SVM (kernel lineare)']

# Grafico delle curve di accuratezza per il training set
plt.figure(figsize=(10, 6))
plt.plot(accuracies_train, label='Training Set')
plt.xticks(range(len(labels)), labels)
plt.xlabel('Modello SVM')
plt.ylabel('Accuratezza')
plt.title('Curve di Accuratezza sul Training Set')
plt.legend()
plt.show()


# Esegui la validazione incrociata con k ripetizioni (k >= 10)
k = 10
scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')


# Calcola la matrice di confusione
y_pred = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)

# Crea un heatmap della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('Classe Predetta (auto)')
plt.ylabel('Classe Reale (auto)')
plt.title('Matrice di Confusione')
plt.show()



# Calcolo dei valori statistici
mean_score = np.mean(scores)
std_score = np.std(scores)
median_score = np.median(scores)
min_score = np.min(scores)
max_score = np.max(scores)
range_score = max_score - min_score

# Calcolo dell'intervallo di confidenza al 95%
alpha = 0.05
n = len(scores)
mean_std = std_score / np.sqrt(n)
t_value = 2.262  # Valore critico per k=10, alpha=0.05 (consultare tabella t-student)
lower_bound = mean_score - t_value * mean_std
upper_bound = mean_score + t_value * mean_std

# Creazione istogramma
plt.hist(scores, bins=10, edgecolor='black')
plt.xlabel('Punteggio')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei punteggi')
plt.show()

# Creazione del grafico scatter
plt.scatter(range(len(scores)), scores)
plt.xlabel('Indice del campione')
plt.ylabel('Punteggio')
plt.title('Distribuzione dei punteggi')
plt.show()

# Stampa dei risultati
print(f"Media: {mean_score:.4f}")
print(f"Mediana: {median_score:.4f}")
print(f"Minimo: {min_score:.4f}")
print(f"Massimo: {max_score:.4f}")
print(f"Intervallo: {range_score:.4f}")
print(f"Deviazione standard: {std_score:.4f}")
print(f"Intervallo di confidenza al 95%: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Inizializzazione dei dati
tempo = np.arange(10)  # Esempio di tempo da 0 a 9
metrica1 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])# Esempio per la metrica 1
metrica2 = np.array([5, 5, 5, 5, 5, 10, 10, 10, 10, 10])# Esempio per la metrica 2
metrica3 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])# Esempio per la metrica 3

# Creazione del plot grafico
plt.figure(figsize=(10, 6))
plt.plot(tempo, metrica1, label='Metrica 1')
plt.plot(tempo, metrica2, label='Metrica 2')
plt.plot(tempo, metrica3, label='Metrica 3')
plt.xlabel('Tempo')
plt.ylabel('Valore')
plt.title('Andamento delle metriche nel tempo')
plt.legend()
plt.grid(True)
plt.show()







X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)


# Definisci il numero di feature
n_features = X_train.shape[1]


# Definisci il modello
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(n_features,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compila il modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Definisci il callback di TensorBoard
tensorboard_callback = TensorBoard(log_dir='./logs')

print("Sto utilizzando il callback di TensorBoard.")

# Addestra il modello
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Avvia TensorBoard per visualizzare i risultati
#%load_ext tensorboard
#%tensorboard --logdir=./logs




