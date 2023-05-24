from itertools import combinations
import math
import bisect
import sys


def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in combinations(List, i + 1)]
    return PS


def shap(characteristic_function, n):
    tempList = list([i for i in range(n)])
    N = power_set(tempList)
    shapley_values = []
    for i in range(n):
        shapley = 0
        for j in N:
            if i not in j:
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui, i)
                l = N.index(j)
                k = N.index(Cui)
                temp = float(float(characteristic_function[k]) - float(characteristic_function[l])) * \
                       float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = N.index(Cui)
        temp = float(characteristic_function[k]) * float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(
            math.factorial(n))
        shapley += temp

        shapley_values.append(shapley)

    return shapley_values


# if __name__ == '__main__':
#     characteristic_function = ['1', ' 3', ' 4', ' 4', '5', ' 8', '10']
#     n = 3
#     shap(characteristic_function, n)
