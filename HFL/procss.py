
# divide the data by given precentsge no overlaps
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

# divide the dsta by gender no overlap
def divide_data_with_gender(data, prc: list, bank:bool) -> list:
    # different spelling and numerical value for bank or diabetes data
    if bank:
        male_data = data[data['Gender'] == 1]
        female_data = data[data['Gender'] == 0]
    else:
        male_data = data[data['gender'] == 1]
        female_data = data[data['gender'] == -1]

    participant_1_samples = int(len(data) * prc[0])
    participant_2_samples = int(len(data) * prc[1])

    participant_1_data = male_data.sample(n=participant_1_samples)
    participant_2_data = female_data.sample(n=participant_2_samples)
    participant_3_data = data.drop(participant_1_data.index).drop(participant_2_data.index)

    # print(len(participant_1_data),len(participant_2_data),len(participant_3_data))
    # print(set(participant_1_data['Gender']))
    # print(set(participant_2_data['Gender']))
    # print(set(participant_3_data['Gender']))
    return [participant_1_data,participant_2_data,participant_3_data]
