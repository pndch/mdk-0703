from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y); plt.show()

# сигмоида преобразует любое число в диапазон от 0 до 1
#sig = lambda z: 1 / (1 + np.exp(-z))
tanh = lambda z: np.tanh(z)

rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0 # w - вектор весов b - биас
def forward(X): 
    z = X @ w + b
    return (np.tanh(z) + 1) / 2#sig(X @ w + b)
print(((forward(X) > .5) == y).mean())

# функция потерь, p - предсказания y - истинные метки классов
def bce(p, y): return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
print('bce: ', bce(forward(X), y))

# градиентный спуск
lr, hist = 0.1, [] # lr - learning rate, hist - список для сохранения значений bce
for i in range(500):
    p = forward(X) # модель вычисляет текущие вероятности для точек             
    dz = p - y # вычисление ошибки (градиента)
    w -= lr * X.T @ dz / len(X) # шаг градиентного спуска
    b -= lr * dz.mean() # обновление смещения (bias)
    hist.append(bce(p, y))
plt.plot(hist); plt.show()
print('accuracy:', ((forward(X) > .5) == y).mean())#((forward(X) > .5) == y).mean())

# деление набора данных на обучающую и контрольную выборку и визуализация
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, zz, levels=[0, .5, 1], alpha=.3); plt.scatter(X[:,0], X[:,1], c=y); plt.show()

