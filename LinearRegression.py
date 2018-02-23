import numpy as np

class LinearRegression:

    def __init__(self, X, y, learning_rate=0.001, lbda=0):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.learning_rate = learning_rate
        self.lbda = lbda

    def normal_equation(self):
        first_part = np.linalg.inv((self.lbda * np.eye(self.d)) + np.dot(np.transpose(self.X), self.X))
        second_part = np.dot(np.transpose(self.X), self.y)
        self.w = np.dot(first_part, second_part)

    def SSE_gradient(self):
        first_part = np.transpose(self.X)
        second_part = np.dot(self.X, self.w) - self.y
        grad = 2 * np.dot(first_part, second_part)
        return grad

    def fit(self):
        self.normal_equation()
