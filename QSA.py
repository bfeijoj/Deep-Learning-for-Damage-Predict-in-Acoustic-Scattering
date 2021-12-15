import numpy as np

class Stiffness:
    
    def __init__(self, thickness, rho, P_Speed, S_Speed):
        self.thickness = thickness
        self.rho = rho
        self.P_Speed = P_Speed
        self.S_Speed = S_Speed
    
    def calculate(self, defect = 0):
        self.defect = defect
        mu = (self.S_Speed ** 2) * self.rho
        lamb = (self.P_Speed ** 2) * self.rho - 2 * mu
        s = 1 / mu
        c = 2 * mu + lamb
        if defect > 0:
            K = (1 - self.defect) * np.linalg.inv(self.thickness * np.array([[s, 0, 0], [0, s, 0], [0, 0, 1 / c]]))
            return K
        K = np.linalg.inv(self.thickness * np.array([[s, 0, 0], [0, s, 0], [0, 0, 1 / c]]))
        return K