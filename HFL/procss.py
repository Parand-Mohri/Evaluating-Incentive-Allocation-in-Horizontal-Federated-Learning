
def divide_data(data, prc: list) -> list:
    s = len(data)
    cont = []
    for i in range(0,len(prc)):
        # print('here')
        if i == 0:
            print(s)
            # print(round(s*prc[i]))
            cont.append(data.iloc[:round(s*prc[i])])
        else:
            # print(round(s*(prc[i-1] + prc[i]) - round(s*prc[i-1])))
            cont.append(data.iloc[round(s*prc[i-1]): round(s*(prc[i-1] + prc[i]))])
    # print(len(cont[0]), len(cont[1]), len(cont[2]))
    # print(len(cont[0]), len(cont[1]))
    return cont

