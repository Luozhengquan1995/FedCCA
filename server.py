import numpy as np
from numpy.linalg import svd
from client import Client
import pdb

class Server:
    def __init__(self, client_list):
        self.client_list = client_list
        self.num_clients = len(client_list)
        self.lines = [client.get_sample_num() for client in self.client_list]
        self.beginend=[]
        start=0
        for client in self.client_list:
            self.beginend.append([start, start+self.lines[client.idx]])
            start+=self.lines[client.idx]            

    def calculate(self, A, name='X', option='left', dp=False, client_ratio=1.0):
        num_selected_clients = int(client_ratio * self.num_clients)
        selected_clients = np.random.choice(self.client_list, num_selected_clients, replace=False)
        selected_clients = sorted(selected_clients, key=lambda client: client.idx)
        #selected_clients = self.client_list
        results = []
        line = 0
        for client in selected_clients:
            #pdb.set_trace()
            #print(client.idx)
            if name == 'X_T' or name == 'Y_T':
                result = client.calculate(A, name, option, dp)
                if len(results) == 0:
                    results = result
                else:
                    results = np.concatenate((results, result), axis=0) 
            else:
                #pdb.set_trace()
                if len(A.shape)==1:
                    A_splits=A[line:line+client.get_sample_num(),]
                else:
                    A_splits=A[line:line+client.get_sample_num(),:]
                result = client.calculate(A_splits, name, option, dp) 
                if len(results) == 0:
                    results = result
                else:
                    results += result
                line=line+client.get_sample_num()
        # Aggregate results (e.g., averaging)
        return results
        
    def get_sample_num(self):
        num=0
        for client in self.client_list:
            num += client.get_sample_num()
        return num
    def get_dx(self):
        return self.client_list[0].get_dx()
    def get_dy(self):
        return self.client_list[0].get_dy()
        
# Example usage
if __name__ == "__main__":
    # Example data for the clients
    num_clients = 5
    client_list = []
    for i in range(num_clients):
        X = np.random.rand(100, 10)
        Y = np.random.rand(100, 5)
        sigma = 0.5
        client1 = Client(X, Y, idx=i, sigma=sigma)
        client_list.append(client1)

    server = Server(client_list)
    A = np.random.rand(10, 5)  # Example A matrix
    aggregated_result = server.calculate(A, name='X', option='left', dp=True, client_ratio=0.6)
    
    print("Aggregated Result:", aggregated_result)
    print("Total Sample Number:", server.get_sample_num())