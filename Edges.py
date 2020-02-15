import numpy as np

class Edges:
    def __init__(self, dim_to, dim_from):
        self.weights = np.random.rand(dim_to, dim_from)