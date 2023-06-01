def bp(estate, claims):
    total_claim = sum(claims)
    if estate >= total_claim:
        print("estate is enough for all the claims")
        return
    payoff = []
    number_of_people = len(claims)
    for cl in claims[:-1]:
        c1 = cl
        c2 = total_claim - cl
        p1 = c1/2
        p2 = estate - p1
        if not test_1(number_of_people-1,p1,p2):
            # print("test 1 failed")
            index = len(claims) - number_of_people
            p = estate / (len(claims) - index)
            while len(payoff) < len(claims):
                payoff.append(p)
            print(payoff)
            return
        if not test_2(number_of_people-1, p1, p2, c2):
            # print("test 2 failed")
            index = len(claims) - number_of_people
            loss = (total_claim - estate) / (len(claims) - index)
            while index < len(claims):
                payoff.append(claims[index] - loss)
                index += 1
            print(payoff)
            return

        payoff.append(p1)
        total_claim -= cl
        estate = p2
        number_of_people -= 1

    payoff.append(p2)
    print(payoff)

def test_1(number_of_people, p1, p2):
#     # enough in system for everyone to get p1
    if number_of_people * p1 <= p2:
        return True
    else:
        return False
#
def test_2(number_of_people, p1, p2, c2 ):
#     # everyone loosing as much as p1
    if number_of_people * p1 <= c2 - p2:
        return True
    else:
        return False
