import numpy as np

def cgls(A, b):
    height, width = A.shape
    x = np.zeros((height))
    # print(A)
    while(True):
        sumA = A.sum()
        # print("sum: " + str(np.absolute(sumA)))
        # print("det: " + str(np.absolute(np.linalg.det(A))))
        if (sumA < 100):
            break
        if (np.linalg.det(A) < 1):
            A = A + np.eye(height, width) * sumA * 0.000000005
        else:
            x = np.linalg.inv(A).dot(b)
            break
    return x
