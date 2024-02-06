import math

import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns


def koeff1(x):
    return x - math.cos((0.7854 - x * math.sqrt(1 - x * x)) / (1 + 2 * x * x))


def funkoeff2(x):
    return math.sqrt(x) - math.cos(x)


def intkoeff2(x):
    return math.sin(x) / (math.cos(x) * math.cos(x))


def rkfun(t, dx, h, a, n):
    x = np.zeros(n)
    for i in range(1, n - 1):
        x[i] = a * (dx[i - 1] - 2 * dx[i] + dx[i + 1]) / h ** 2
    return x


def rk(a, n, t, t0, l):
    h = l / n
    starttemp = np.zeros(n)
    for i in range(n):
        hi = i * h
        print(hi)
        if (0.4 - 0.001) <= hi <= (0.7 + 0.001):
            starttemp[i] = 5
    teval = np.linspace(t0, t, num=n)
    sol = integrate.solve_ivp(rkfun, [t0, t], starttemp, t_eval=teval, rtol=0.00001, atol=0.0001, args=(h, a, n))
    temp = sol.y
    return temp


def t(a, t_end, l0, hprint):
    n = int(t_end / hprint)
    h = l0 / n
    tau = h ** 2 / (2 * a)
    m = int(t_end / tau)
    temp = np.zeros((m + 1, n + 1))
    time = np.zeros(m)
    for i in range(m):
        time[i] = tau * i
        temp[i][0] = 0
        temp[i][l0] = 0
    length = np.zeros(n)
    for i in range(n):
        length[i] = i * h
        if 0.4 <= length[i] <= 0.7:
            temp[0][i] = 5
    for i in range(0, m - 1):
        for j in range(1, n - 1):
            temp[i + 1][j] = 1 / 2 * (temp[i][j + 1] + temp[i][j - 1])
    temp_print = np.zeros((n, n))
    length_print = np.zeros(n)
    time_print = np.zeros(n)
    l_print = l0 / n
    t_print = hprint
    for i in range(n):
        time_print[i] = t_print * i
        length_print[i] = l_print * i
    for j in range(n):
        for i in range(m):
            if abs(time[i] - time_print[j]) < 0.001:
                for k in range(n):
                    temp_print[j][k] = temp[i][k]
                break

    return temp_print, length_print, time_print


k1 = optimize.brentq(koeff1, 0, 1, xtol=0.000001, rtol=0.000001)
print("k1 = ", k1)
x_ = optimize.minimize_scalar(funkoeff2, bounds=(5, 8), method='bounded')
print("x* = ", x_.x)
y_, err_ = integrate.quad(intkoeff2, 0, math.pi / 4)
print("y* = ", y_)
k2 = x_.x * y_
print("k2 = ", k2)
print("error 1 = 0.000001", '', "error 2 = ", err_)

temp1, length1, time1 = t(k1, 3, 1, 0.05)
Y1, X1 = np.meshgrid(length1, time1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Temperature 1')
my_cmap = plt.get_cmap('viridis')
sur = ax.plot_surface(X1, Y1, temp1, cmap=my_cmap, edgecolor='none')
fig.colorbar(sur, ax=ax, shrink=0.7, aspect=7)
ax.view_init(15, 70)
plt.show()

temp2, length2, time2 = t(k2, 3, 1, 0.05)
Y2, X2 = np.meshgrid(length2, time2)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Temperature 2')
sur = ax.plot_surface(X2, Y2, temp2, cmap = my_cmap, edgecolor ='none')
fig.colorbar(sur, ax = ax, shrink = 0.7, aspect = 7)
ax.view_init(15, 70)
plt.show()

Temp1 = rk(k1, 60, 3, 0, 1)
Y1, X1 = np.meshgrid(length1, time1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Temperature 1 in rk')
my_cmap = plt.get_cmap('viridis')
sur = ax.plot_surface(Y1, X1, Temp1, cmap=my_cmap, edgecolor='none')
fig.colorbar(sur, ax=ax, shrink=0.7, aspect=7)
ax.view_init(15, 70)
plt.show()

Temp2 = rk(k2, 60, 3, 0, 1)
Y1, X1 = np.meshgrid(length1, time1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Temperature 2 in rk')
my_cmap = plt.get_cmap('viridis')
sur = ax.plot_surface(Y1, X1, Temp2, cmap=my_cmap, edgecolor='none')
fig.colorbar(sur, ax=ax, shrink=0.7, aspect=7)
ax.view_init(15, 70)
plt.show()
