import numpy as np

a = np.array([3, 1, 4, 1, 5])
print(a * 2) #умножение
print(a.mean(), a.min(), a.max(), a.sum())

M = np.arange(12).reshape(3, 4) #заполняет числами последовательно от 0 до n
print(M, M.shape)
print('строка 0:', M[0])
print('столбец 1:', M[:, 1]) 
print('элемент:', M[2, 3])
print(M.T.shape)

print(M.mean(axis=0)) #строки 
print(M.mean(axis=1)) #столбцы

A = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1], [0], [-1]])
print(A @ w)                        #перемножение матриц
print(A + np.array([10, 20, 30]))   #сложение матриц

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5]); b = 0.3
y = X @ w + b
print(X.shape, w.shape, y.shape, y[:5])


with open('data/titanic.csv', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]

cols = {h: [] for h in header} #список ключ значение с заголовками

for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)

a = np.array(cols['age'])

print(np.median(a), a.mean(), a.min(), a.max(), a.sum())