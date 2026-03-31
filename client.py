import numpy as np
from numpy.linalg import svd

class Client:
    def __init__(self, X, Y, idx, sigma):
        self.X = X
        self.Y = Y
        self.idx = idx
        self.sigma = sigma

    def calculate(self, A, name='X', option='left', dp=False):
        if option=='left':
            if name=='X':
                out = self.X @ A
            elif name=='X_T':
                out = self.X.T @ A
            elif name == 'Y':
                out = self.Y @ A
            elif name=='Y_T':
                out = self.Y.T @ A
            else:
                print("wrong name")
        elif option=='right':
            if name=='X':
                out = A @ self.X
            elif name=='X_T':
                out = A @ self.X.T
            elif name == 'Y':
                out = A @ self.Y
            elif name=='Y_T':
                out = A @ self.Y.T
            else:
                print("wrong name")
        else:
            print("wrong option")
        
        if dp==True:
            out=self.apply_differential_privacy(out)
            
        return out

    def apply_differential_privacy(self, result):
        # Adding Gaussian noise to the result to ensure differential privacy
        noise = np.random.normal(0, self.sigma, size=result.shape)
        private_result = result + noise
        return private_result
    def get_sample_num(self):
        return self.X.shape[1]
        
    def get_dx(self):
        return self.X.shape[0]
    def get_dy(self):
        return self.Y.shape[0]
# Example usage
if __name__ == "__main__":
    # Example data for the client
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 5)
    A = np.random.rand(10, 5)
    sigma = 0.5

    client = Client(X, Y, idx=1, sigma=sigma)
    result = client.calculate(A, name='X', option='left', dp=False)
    private_result = client.calculate(A, name='X', option='left', dp=True)
    
    print("Result:", result)
    print("Private Result:", private_result)