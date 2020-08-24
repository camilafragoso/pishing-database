import pandas as pd
import seaborn as sns
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Abrindo o arquivo csv
base = pd.read_csv('data.csv')

#Descrição da base
base.describe()

#Número de exemplos por classe
suspicious=len(base[base.Result==0])
legitimate=len(base[base.Result==1])
pishing=len(base[base.Result==-1])

print(suspicious, 'suspicious activities found')
print(pishing, 'pishing activities found')
print(legitimate, 'legitimate activities found')

#Gráfico do número de exemplos por classe
sns.countplot(base['Result'])

#Checando se existem dados faltantes
base.info()

#Note que existem 1353 linhas, então não existem dados nulos de acordo com as informações obtidas acima. 
base.shape

# definindo os previsores e a classe
predictors = base.iloc[:, 0:9].values
classe = base.iloc[:, 9].values

# dividindo a base em treinamento e teste
from sklearn.model_selection import train_test_split
predictors_training, predictors_test, classe_training, classe_test = train_test_split(predictors, classe, test_size=0.25, random_state=0)

# GAUSSIAN NB
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(predictors_training, classe_training)
predictions_gaussianNB = classifier1.predict(predictors_test)

#ACURÁCIA
accuracy_gaussianNB = accuracy_score(classe_test, predictions_gaussianNB)
print(accuracy_gaussianNB)

#F1 SCORE
f1_score_gaussian = f1_score(classe_test, predictions_gaussianNB, average='weighted')
print(f1_score_gaussian)

#RECALL
recall_gaussian = recall_score(classe_test, predictions_gaussianNB, average='weighted')
print(recall_gaussian)

#PRECISÃO
precision_gaussian = precision_score(classe_test, predictions_gaussianNB, average='weighted')
print(precision_gaussian)

# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier()
classifier2.fit(predictors_training, classe_training)
predictions_decisionTree = classifier2.predict(predictors_test)

#ACURÁCIA
accuracy_decisionTree = accuracy_score(classe_test, predictions_decisionTree)
print(accuracy_decisionTree)

#F1 SCORE
f1_score_decisionTree = f1_score(classe_test, predictions_decisionTree, average='weighted')
print(f1_score_decisionTree)

#RECALL
recall_decisionTree = recall_score(classe_test, predictions_decisionTree, average='weighted')
print(recall_decisionTree)

#PRECISÃO
precision_decisionTree = precision_score(classe_test, predictions_decisionTree, average='weighted')
print(precision_decisionTree)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression()
classifier3.fit(predictors_training, classe_training)
predictions_logisticRegression = classifier3.predict(predictors_test)

#ACURÁCIA
accuracy_logisticRegression = accuracy_score(classe_test, predictions_logisticRegression)
print(accuracy_logisticRegression)

#F1 SCORE
f1_score_logisticRegression = f1_score(classe_test, predictions_logisticRegression, average='weighted')
print(f1_score_logisticRegression)

#RECALL
recall_logisticRegression = recall_score(classe_test, predictions_logisticRegression, average='weighted')
print(recall_logisticRegression)

#PRECISÃO
precision_logisticRegression = precision_score(classe_test, predictions_logisticRegression, average='weighted')
print(precision_logisticRegression)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier()
classifier4.fit(predictors_training, classe_training)
predictions_KNN = classifier4.predict(predictors_test)

#ACURÁCIA
accuracy_KNN = accuracy_score(classe_test, predictions_KNN)
print(accuracy_KNN)

#F1 SCORE
f1_score_KNN = f1_score(classe_test, predictions_KNN, average='weighted')
print(f1_score_KNN)

#RECALL
recall_KNN = recall_score(classe_test, predictions_KNN, average='weighted')
print(recall_KNN)

#PRECISÃO
precision_KNN = precision_score(classe_test, predictions_KNN, average='weighted')
print(precision_KNN)

#SVC
from sklearn.svm import SVC
classifier5 = SVC(C=8.0)
classifier5.fit(predictors_training, classe_training)
predictions_svc = classifier5.predict(predictors_test)

#ACURÁCIA
accuracy_svc = accuracy_score(classe_test, predictions_svc)
print(accuracy_svc)

#F1 SCORE
f1_score_svc = f1_score(classe_test, predictions_svc, average='weighted')
print(f1_score_svc)

#RECALL
recall_svc = recall_score(classe_test, predictions_svc, average='weighted')
print(recall_svc)

#PRECISÃO
precision_svc = precision_score(classe_test, predictions_svc, average='weighted')
print(precision_svc)



#SALVANDO RESULTADOS EM CSV - formato para leitura
with open('read_results.csv', 'w', newline='') as file:
    fieldnames = ['classifier', 'accuracy', 'precision', 'recall', 'f1']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'classifier': 'GaussianNB', 'accuracy': accuracy_gaussianNB, 'precision': precision_gaussian, 'recall': recall_gaussian, 'f1': f1_score_gaussian})
    writer.writerow({'classifier': 'Decision Tree', 'accuracy': accuracy_decisionTree, 'precision': precision_decisionTree, 'recall': recall_decisionTree, 'f1': f1_score_decisionTree})
    writer.writerow({'classifier': 'Logistic Regression', 'accuracy': accuracy_logisticRegression, 'precision': precision_logisticRegression, 'recall': recall_logisticRegression, 'f1': f1_score_logisticRegression})
    writer.writerow({'classifier': 'KNN', 'accuracy': accuracy_KNN, 'precision': precision_KNN, 'recall': recall_KNN, 'f1': f1_score_KNN})
    writer.writerow({'classifier': 'SVC', 'accuracy': accuracy_svc, 'precision': precision_svc, 'recall': recall_svc, 'f1': f1_score_svc})
    
#Gerando a tabela com os dados obtidos
results_readable = pd.read_csv('read_results.csv')
results_readable

#SALVANDO RESULTADOS EM CSV - formato para plotar o gráfico
with open('graphic_result.csv', 'w', newline='') as file:
    fieldnames = ['metric', 'value', 'classifier']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()              
    writer.writerow({'metric': 'accuracy', 'value': accuracy_gaussianNB, 'classifier': 'GaussianNB'})
    writer.writerow({'metric': 'accuracy', 'value': accuracy_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'accuracy', 'value': accuracy_logisticRegression, 'classifier': 'Logistic Regression'})
    writer.writerow({'metric': 'accuracy', 'value': accuracy_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'accuracy', 'value': accuracy_svc, 'classifier': 'SVC'})
    writer.writerow({'metric': 'precision', 'value': precision_gaussian, 'classifier': 'GaussianNB'})
    writer.writerow({'metric': 'precision', 'value': precision_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'precision', 'value': precision_logisticRegression, 'classifier': 'Logistic Regression'})
    writer.writerow({'metric': 'precision', 'value': precision_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'precision', 'value': precision_svc, 'classifier': 'SVC'})
    writer.writerow({'metric': 'f1', 'value': f1_score_gaussian, 'classifier': 'GaussianNB'})
    writer.writerow({'metric': 'f1', 'value': f1_score_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'f1', 'value': f1_score_logisticRegression, 'classifier': 'Logistic Regression'})
    writer.writerow({'metric': 'f1', 'value': f1_score_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'f1', 'value': f1_score_svc, 'classifier': 'SVC'})
    writer.writerow({'metric': 'recall', 'value': recall_gaussian, 'classifier': 'GaussianNB'})
    writer.writerow({'metric': 'recall', 'value': recall_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'recall', 'value': recall_logisticRegression, 'classifier': 'Logistic Regression'})
    writer.writerow({'metric': 'recall', 'value': recall_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'recall', 'value': recall_svc, 'classifier': 'SVC'})

#Tabela formatada para a geração do gráfico
formated = pd.read_csv('graphic_result.csv')
formated

#Gráfico comparativo de acurácia, precisão, f1 e recall entre os classificadores
sns.catplot(x='metric', y='value', hue='classifier', data=formated, kind='bar')

#SALVANDO melhores RESULTADOS EM CSV - formato para plotar o gráfico

with open('best_graphic_result.csv', 'w', newline='') as file:
    fieldnames = ['metric', 'value', 'classifier']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()      
    writer.writerow({'metric': 'accuracy', 'value': accuracy_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'accuracy', 'value': accuracy_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'accuracy', 'value': accuracy_svc, 'classifier': 'SVC'})
    writer.writerow({'metric': 'precision', 'value': precision_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'precision', 'value': precision_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'precision', 'value': precision_svc, 'classifier': 'SVC'})
    writer.writerow({'metric': 'f1', 'value': f1_score_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'f1', 'value': f1_score_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'f1', 'value': f1_score_svc, 'classifier': 'SVC'})
    writer.writerow({'metric': 'recall', 'value': recall_decisionTree, 'classifier': 'Decision Tree'})
    writer.writerow({'metric': 'recall', 'value': recall_KNN, 'classifier': 'KNN'})
    writer.writerow({'metric': 'recall', 'value': recall_svc, 'classifier': 'SVC'})
    
#Abrindo o arquivo com os melhores resultados
best_results = pd.read_csv('best_graphic_result.csv')

#Plotando o gráfico dos melhores resultados
sns.catplot(x='metric', y='value', hue='classifier', data=best_results, kind='bar')



# EXTRA - MATRIZ CONFUSÃO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

matriz_gaussianNB = confusion_matrix(classe_test, predictions_gaussianNB)
print(matriz_gaussianNB)

matriz_logisticRegression = confusion_matrix(classe_test, predictions_logisticRegression)
print(matriz_logisticRegression)

matriz_decisionTree = confusion_matrix(classe_test, predictions_decisionTree)
print(matriz_decisionTree)


#TO READ 
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html