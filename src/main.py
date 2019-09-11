#
# IMPORTS
#
import scipy.io as sio
import numpy as np


#
# CODE
#
def GP(a, r, n):
    gp_list = list()

    for i in range(0, n):  
        curr_term = a * pow(r, i)
        gp_list.append(curr_term)
 
    return gp_list


def main():
    # get data
    data = sio.loadmat('data.mat')

    X = data['X']
    Y = data['S']
    offset = np.ones((60000, 1))

    # add offset to matrix X
    x_offset = np.append(offset, X, axis = 1)

    # set coeficientes
    coefs = GP(pow(2,-10), pow(2,2), 13)

    # set eye matrix
    eye = np.eye(785)

    # get 40000 samples to train
    x_train = x_offset[:40000]
    y_train = Y[:40000]

    # get 20000 samples for x_validation
    x_validation = x_offset[40000:60000]
    y_validation = Y[40000:60000]

    classification_percentage = []

    # for each coeficiente: get quadratic error and classification error
    for coef in coefs:

        # make operations to get W matrix: W = (Xt*X + lambda * I)-1 * Xt * y
        op1 = x_train.transpose().dot(x_train)
        op2 = coef * eye
        op3 = op1 + op2
        op4 = np.linalg.inv(op3)
        op5 = op4.dot(x_train.transpose())
        W = op5.dot(y_train)

        # get validation
        output = x_validation.dot(W)

        # initialize matches
        matches = 0

        # for each output: check if it was a miss or not and calculate error
        for i in range(0,20000):
            if (np.argmax(output[i]) == np.argmax(y_validation[i])):
                matches = matches + 1

        accuraccy = float(matches) / float(20000)

        classification_percentage.append(accuraccy)

    print(coefs)
    print(classification_percentage)

    

if __name__ == "__main__":
    main()
