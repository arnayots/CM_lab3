import sys

import matplotlib.pyplot as plt
import numpy as np
import math

a = -2
b = 3

debug = False
#debug = True
printPlots = True
printPlots = False

def f(x):
    return 0.5 * (abs(x**2 - 1) - abs(x)) + 1

def f_ort(t):
    x = 2.5 * t / math.pi - 2.5
    return f(x)

def get_equal_pts(n):
    tmp = list([])
    h = (b - a) / n
    for i in range(0, n+1):
        tmp.append(a + i * h)
    if debug:
        print(tmp)
    return tmp

def get_chebyshev_pts(n):
    tmp = list([])
    for i in range(0, n+1):
        tmp.append((a+b) * 0.5 + 0.5 * (b-a) * math.cos(math.pi * i / (n + 1)))
    if debug:
        print(tmp)
    return tmp

def interpolate(n, xi):
    matr = np.empty((n, n))
    for i in range(0, n):
        for j in range(0, n):
            matr[i, j] = 0

    for i in range(0, n):
        matr[i, 0] = f(xi[i])
    if debug:
        print(matr)
        print(xi)

    for j in range(1, n):
        for i in range(0, n - j):
            matr[i, j] = (matr[i + 1, j - 1] - matr[i, j - 1]) / (xi[i + j] - xi[i])
    if debug:
        print(matr)
    return matr

def print_table(n, xi, matr):
    print("----- Таблиця розділених різниць -----")
    for i in range(0, n):
        s = "" + str(xi[i]) + "   "
        for j in range(0, n - i):
            s += str(matr[i, j]) + "   "
        print(s)

def get_str_newton(n, xi, coef):
    res = ""
    for i in range(0, n):
        tmp = ""
        if(coef[i] > 0):
            tmp += "+ "
        else:
            tmp += "- "
        tmp += str(abs(coef[i]))
        for j in range(0, i):
            if(xi[j] != 0):
                tmp += " * (x "
                if(xi[j] < 0):
                    tmp += "+ "
                else:
                    tmp += "- "
                tmp += str(abs(xi[j])) + ")"
            else:
                tmp += " * x"
        res += tmp + " "
    return res

def get_coefs(matr, n):
    coef = list([])
    for i in range(0, n):
        coef.append(matr[0, i])
    return coef

def eval_newton(x, n, xi, coef):
    res = 0
    for i in range(0, n):
        tmp = coef[i]
        for j in range(0, i):
            tmp *= (x - xi[j])
        res += tmp
    return res

def h(i, xi):
    return xi[i] - xi[i - 1]

def get_zero_matr(n, m):
    matr = np.empty((n, m))
    for i in range(0, n):
        for j in range(0, m):
            matr[i, j] = 0
    return matr

def get_spline_coefs(n, xi):
    y = list([])
    for i in range(0, n):
        y.append(f(xi[i]))

    matrA = get_zero_matr(n, n)
    vecB = list([0] * n)
    for i in range(1, n - 1):
        matrA[i - 1, i - 1] = h(i, xi) / 6
        matrA[i - 1, i] = (h(i, xi) + h(i + 1, xi)) / 3
        matrA[i - 1, i + 1] = h(i + 1, xi) / 6
        vecB[i - 1] = (y[i + 1] - y[i]) / h(i + 1, xi) - (y[i] - y[i - 1]) / h(i, xi)
    matrA[n - 2, 0] = 1
    matrA[n - 1, n - 1] = 1

    return np.linalg.solve(matrA, vecB)

def get_sign(n):
    if n < 0:
        return "-"
    else:
        return "+"

def near(x, y):
    return abs(x - y) < 1E-8

def get_str_spline(n, m, xi, y):
    res = ""
    for i in range(1, n):
        tmp = "S(x) = "
        tmp2 = m[i - 1] / (6 * h(i, xi))
        if not near(tmp2, 0):
            tmp += str(tmp2) + " * ("+ str(xi[i]) +" - x)^3 "
        tmp2 = m[i] / (6 * h(i ,xi))
        if not near(tmp2, 0):
            tmp += get_sign(tmp2) + " " + str(abs(tmp2)) + " * (x " + get_sign(xi[i-1]) + " " + str(abs(xi[i-1])) + ")^3 "
        tmp2 = (y[i] - (m[i] * (h(i, xi)**2) / 6)) / h(i, xi)
        if not near(tmp2, 0):
            tmp += get_sign(tmp2) + " " + str(abs(tmp2)) + " * (x "+ get_sign(xi[i]) + " "+ str(abs(xi[i])) +") "
        tmp2 = (y[i-1] - (h(i, xi)**2) * m[i-1] / 6) / h(i, xi)
        if not near(tmp2, 0):
            tmp += get_sign(tmp2) + " " + str(abs(tmp2)) + " * (" + str(xi[i]) + " - x)"
        tmp += "\n x = ["+ str(xi[i - 1]) +" , "+ str(xi[i]) +"]\n"
        res += tmp
    return res

def spline(x_var: float, n, m, xi, y):
    tmp = list([])
    for k in range(0, len(x_var)):
        x = x_var[k]
        res = 0
        i = -1
        for j in range(1, n):
            if (x >= xi[j - 1]) and (x <= xi[j]):
                i = j
        if(i != -1):
            res = m[i-1] / (6 * h(i, xi)) * ((xi[i] - x)**3)
            res += m[i] / (6 * h(i ,xi)) * ((x - xi[i-1])**3)
            res += ((y[i] - (m[i] * (h(i, xi)**2) / 6)) / h(i, xi)) * (x - xi[i-1])
            res += ((y[i-1] - (h(i, xi)**2) * m[i-1] / 6) / h(i, xi)) * (xi[i] - x)
        tmp.append(res)
    return tmp

def integrate(func, a, b, N):
    sum = 0
    h = (b - a) / N
    for i in range(0, N+1):
        sum += h * func(a + i*h - h/2)
    return sum

def phi_1(i, x):
    return x**i

def phi_2(i, x):
    if i == 0:
        return 1
    if i % 2 == 0:
        return math.sin(x * i * 0.5)
    else:
        return math.cos(x * (i + 1) * 0.5)

def str_phi_1(vec, n):
    res = str(vec[0])
    for i in range(1, n):
        res += " + " + str(vec[i]) + " * x^" + str(i)
    return res

def val_phi_1(vec, n, x):
    res = vec[0]
    for i in range(1, n):
        res += vec[i] * x**i
    return res

def str_phi_2(vec, n):
    res = str(vec[0])
    for i in range(1, n):
        if i % 2 == 0:
            res += " + " + str(vec[i]) + " * sin(" + str(i/2) + " * x)"
        else:
            res += " + " + str(vec[i]) + " * cos(" + str((i+1)/2) + " * x)"
    return res

def val_phi_2_sc(vec, n, x):
    res = 0
    for i in range(0, n):
        res += vec[i] * phi_2(i, x)
    return res

def val_phi_2(vec, n, x):
    res = []
    for j in range(0, len(x)):
        tmp = val_phi_2_sc(vec, n, x[j])
        res.append(tmp)
    return res

if __name__ == '__main__':
    print("----- PART 1 -----")

    segments = 19
    n = segments + 1

    xi = get_equal_pts(segments)
    print("Точки розподілу:")
    print(xi)
    matr = interpolate(n, xi)
    print_table(n, xi, matr)

    coef_1 = get_coefs(matr,n)
    print("Поліном:")
    print(get_str_newton(n, xi, coef_1))

    xi2 = get_chebyshev_pts(segments)
    print("Точки розподілу:")
    print(xi2)
    matr2 = interpolate(n, xi2)
    print_table(n, xi2, matr2)
    coef_2 = get_coefs(matr2, n)
    print("Поліном:")
    print(get_str_newton(n, xi, coef_2))

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x), color='red', label=r"$f(x)$")
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Функція f(x)')  # назва
        plt.show()

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, eval_newton(x,n,xi,coef_1), color='red', label=r"$Newton 1(Equal distance)$")
        plt.plot(x, eval_newton(x,n,xi2,coef_2), color='green', label=r"$Newton 2(Chebyshev)$")
        plt.plot(x, f(x), color='blue', label=r"$y=f(x)$")
        #axes = plt.gca()
        #axes.set_ylim([-10, 10])
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Поліноми та функція, n = '+str(n))  # назва
        plt.show()

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x) - eval_newton(x, n, xi, coef_1), color='red', label=r"$Newton 1(Equal distance)$")
        plt.plot(x, f(x) - eval_newton(x, n, xi2, coef_2), color='green', label=r"$Newton 2(Chebyshev)$")
        # axes = plt.gca()
        # axes.set_ylim([-10, 10])
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Відхилення поліномів, n = '+str(n))  # назва
        plt.show()

    #    SPLINE
    print("----- SPLINE -----")

    if(n < 3):
        print("Error: Not enough points for spline")
        sys.exit(1)

    y = list([])
    for i in range(0, n):
        y.append(f(xi[i]))
    spl_coefs = get_spline_coefs(n, xi)
    print("Коефіцієенти:")
    print(spl_coefs)
    print(get_str_spline(n, spl_coefs, xi, y))

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x), color='red', label=r"$f(x)$")
        plt.plot(x, spline(x,n,spl_coefs,xi,y), color='green', label=r"$Spline$")
        # axes = plt.gca()
        # axes.set_ylim([-10, 10])
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Сплайн та функція, n = '+str(n))  # назва
        plt.show()

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x) - spline(x,n,spl_coefs,xi,y), color='green', label=r"$Spline difference$")
        # axes = plt.gca()
        # axes.set_ylim([-10, 10])
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Відхилення сплайну), n = '+str(n))  # назва
        plt.show()

    err = max(abs(spl_coefs)) * (b - a) / (n - 1)
    print("Оцінка похибки: ", err)
    print()

    print("----- PART 2 -----")

    n = 20
    # m = 20 Тут фактично N заміняє m з постановки
    N = 1000

    print("Перша множина phi")

    G1 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i + 1):
            G1[i,j] = integrate(lambda x: phi_1(i, x) * phi_1(j, x), a, b, N)
            G1[j,i] = G1[i,j]
    print("Матриця Грама: ")
    print(G1)

    vec_f1 = np.zeros(n)
    for i in range(0, n):
        vec_f1[i] = integrate(lambda x: f(x) * phi_1(i, x), a, b, N)
    print("Вектор f: ")
    print(vec_f1)

    res_1 = np.linalg.solve(G1, vec_f1)
    print("Вектор коефіціентів: ")
    print(res_1)

    print("Поліном найкращого середньоквадратичного наближення: ")
    print(str_phi_1(res_1, n))

    mistake_1 = integrate(lambda x: (f(x) - val_phi_1(res_1, n, x)) ** 2, a, b, N)
    print("Похибка наближення: ")
    print(mistake_1)

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x), color='red', label=r"$f(x)$")
        plt.plot(x, val_phi_1(res_1, n, x), color='green', label=r"$Поліном$")
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Поліном найкращого середньовкадратичного наближення, n = '+str(n))  # назва
        plt.show()

    print("\nДруга множина phi")

    G2 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            G2[i, j] = integrate(lambda x: phi_2(i, x) * phi_2(j, x), a, b, N)
    print("Матриця Грама: ")
    print(G2)

    vec_f2 = np.zeros(n)
    for i in range(0, n):
        vec_f2[i] = integrate(lambda x: f(x) * phi_2(i, x), a, b, N)
    print("Вектор f: ")
    print(vec_f2)

    res_2 = np.linalg.solve(G2, vec_f2)
    print("Вектор коефіціентів: ")
    print(res_2)

    print("Поліном найкращого середньоквадратичного наближення: ")
    print(str_phi_2(res_2, n))

    mistake_2 = integrate(lambda x: (f(x) - val_phi_2_sc(res_2, n, x)) ** 2, a, b, N)
    print("Похибка наближення: ")
    print(mistake_2)

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x), color='red', label=r"$f(x)$")
        plt.plot(x, val_phi_2(res_2, n, x), color='green', label=r"$Поліном$")
        # axes = plt.gca()
        # axes.set_ylim([-10, 10])
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Поліном найкращого середньовкадратичного наближення, n = '+str(n))  # назва
        plt.show()



    print("\nДруга множина phi(Неперервний випадок)")

    res_3 = np.zeros(n)
    for i in range(0,n):
        res_3[i] = integrate(lambda x: f_ort(x) * phi_2(i, x), a, b, N) / integrate(lambda x: phi_2(i, x) * phi_2(j, x), a, b, N)

    print("Вектор коефіціентів: ")
    print(res_3)

    print("Поліном найкращого середньоквадратичного наближення: ")
    print(str_phi_2(res_3, n))

    mistake_3 = integrate(lambda x: (f(x) - val_phi_2_sc(res_2, n, x)) ** 2, a, b, N)
    print("Похибка наближення: ")
    print(mistake_3)

    if printPlots:
        x = np.linspace(a, b, 1000)
        plt.figure()  # створення нового малюнку
        plt.plot(x, f(x), color='red', label=r"$f(x)$")
        plt.plot(x, val_phi_2(res_2, n, x), color='green', label=r"$Поліном$")
        # axes = plt.gca()
        # axes.set_ylim([-10, 10])
        plt.legend(loc='upper left')
        plt.xlabel('x')  # пiдпис осi 0х
        plt.ylabel('y')  # пiдпис осi 0y
        plt.title('Поліном найкращого середньовкадратичного наближення(непер.)\n n = '+str(n))  # назва
        plt.show()

