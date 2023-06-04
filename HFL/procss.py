
def divide_data(data, prc: list) -> list:
    s = len(data)
    cont = []
    for i in range(0,len(prc)):
        if i == 0:
            print(s)
            cont.append(data.iloc[:round(s*prc[i])])
        else:
            cont.append(data.iloc[round(s*prc[i-1]): round(s*(prc[i-1] + prc[i]))])
    return cont

