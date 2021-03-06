import numpy as np
import numpy.random
import math

class ParticleSwarm:
    def __init__(self, cost_fn, dim, size=50, chi=0.72984, phi_p=2.05, phi_g=2.05):
        self.cost_fn = cost_fn
        self.dim = dim

        self.size = size
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.X = np.random.uniform(size=(self.size, self.dim))
        self.V = np.random.uniform(size=(self.size, self.dim))

        self.P = self.X.copy()
        self.S = self.cost_fn(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        

    def optimize(self, epsilon=1e-3, max_iter=100):
        iteration =0
        while self.best_score > epsilon and iteration < max_iter:
            self.update()
            iteration = iteration + 1
        return self.g

    def update(self):
        # Velocities update
        np.random.seed(10)
        R_p = np.random.uniform(size=(self.size, self.dim))
        R_g = np.random.uniform(size=(self.size, self.dim))

        #self.chi = self.chi * 0.786
        
        self.V = (self.chi * self.V) + self.phi_p * R_p * (self.P - self.X)  + self.phi_g * R_g * (self.g - self.X)
                

        # Positions update
        self.X = self.X + self.V
        print("shape:\n",self.X.shape)
        # Best scores
        scores = self.cost_fn(self.X)

##        better_scores_idx = scores < self.S
##        print("index: ",better_scores_idx)
##        self.P[better_scores_idx] = self.X[better_scores_idx]
##        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[scores.argmin()]
        self.best_score = scores.min()
