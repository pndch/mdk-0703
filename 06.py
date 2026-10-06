import numpy as np, matplotlib.pyplot as plt

#
# f(x) = (x − 3)². Производная 2(x − 3) показывает направление роста. Идём против: x = x − lr · f'(x).
#
f = lambda x: (x - 3) ** 2 #целевая функция
df = lambda x: 2 * (x - 3) #ее производная
x, lr, path = 10.0, 0.1, [] #x-стартовая точка lr-скорость обучения path-сюда сохраняется путь
for i in range(30):
    path.append(x)
    x = x - lr * df(x) #проходим по фукнции
print(x)

xs = np.linspace(-2, 12, 100); #массив из равномерных чисел от -2 до 12, размером в 100 элем
# plt.plot(xs, f(xs)); 
# plt.scatter(path, [f(p) for p in path], c='r'); 
# plt.show()

#
# 50 точек y = 2x + 1 + шум. Задача: найти w и b, не зная их.
#
rng = np.random.default_rng(0)
X = rng.uniform(0, 5, 50) #массив из рандомных чисел от 0 до 5, размером в 50 элем
y = 2 * X + 1 + rng.normal(0, 0.5, 50)
# plt.scatter(X, y) 
# plt.show()

def loss(w, b): return np.mean((w * X + b - y) ** 2) #функция потерь (насколько сильно модель ошибается)
def grads(w, b): #вычисление градиентов
    p = w * X + b
    return np.mean(2 * (p - y) * X), np.mean(2 * (p - y))

w, b, lr, hist = 0.0, 0.0, 0.05, []
for i in range(200):
    dw, db = grads(w, b); w -= lr * dw; b -= lr * db; hist.append(loss(w, b))
print(w, b)
plt.plot(hist); plt.show()                                  #ошибки
plt.scatter(X, y); plt.plot(X, w * X + b, 'r'); plt.show()  #предсказания
