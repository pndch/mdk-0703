import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#X - признаки; y - целевой признак
X, y = load_iris(return_X_y=True) 
# разделение на обучающую и контрольную выборку
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0) 
knn = KNeighborsClassifier(3).fit(X_tr, y_tr) #обучение
pred = knn.predict(X_te) #проверка на контрольной выборке
print(pred[:5], y_te[:5], 'accuracy:', accuracy_score(y_te, pred))


df = pd.read_csv('clean.csv')
feats = ['pclass', 'age', 'fare', 'family', 'sex_num']
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df['survived'], test_size=.25, random_state=0)
lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print('accuracy:', lr.score(X_te, y_te), 'baseline:', (y_te == 0).mean())

print(confusion_matrix(y_te, lr.predict(X_te)))
print(classification_report(y_te, lr.predict(X_te)))

new = pd.DataFrame([[1, 25, 80, 0, 1]], columns=feats)
print(lr.predict(new)) #к какому классу модель определила нашего пассажира
print(lr.predict_proba(new)) #вероятность принадлежности к конкретнуму классу из 2 в нашем случае