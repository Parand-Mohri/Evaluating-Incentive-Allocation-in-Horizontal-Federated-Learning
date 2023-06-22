
# Class for the aggregator
class FederatedModel():
    def __init__( self, theta):
        self.theta = theta

    # avrage the weights and biases provided by data owners
    def receive_from_clients(self, theta1, theta2, theta3):
        W = (theta1[0]+theta2[0]+theta3[0])/3
        b = (theta1[1]+theta2[1]+theta3[1])/3
        self.theta = (W,b)

    # use this for experiments with 4 data owners instead of 3
    # def receive_from_clients(self, theta1, theta2, theta3, theta4):
    #     W = (theta1[0]+theta2[0]+theta3[0]+theta4[0])/4
    #     b = (theta1[1]+theta2[1]+theta3[1]+theta4[1])/4
    #     self.theta = (W,b)

    def send_to_clients(self):
        return self.theta