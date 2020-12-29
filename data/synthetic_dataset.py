import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader

def create_sin_dataset(N, N_input, N_output, sigma):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    N_dev = int(0.2 * N)
    N_test = int(0.2 * N)
    num_rolling_windows = 1

    X = []
    breakpoints = []
    for k in range(N):
        inp = np.random.uniform(-np.pi, np.pi) + np.linspace(0, 30, N_input+4*N_output)
        #serie = np.sin(inp)*5 + np.random.normal(0, 0.1, size=(inp.shape)) + 5
        serie = np.sin(inp) + np.random.normal(0, 0.1, size=(inp.shape))
        X.append(serie)
        breakpoints.append(N_input)
    X = np.expand_dims(np.stack(X), axis=-1)
    breakpoints = np.array(breakpoints)

    data_train = []
    data_dev = []
    data_test = []
    dev_tsid_map, test_tsid_map = {}, {}

    for i, ts in enumerate(X):
        entry_dict = {}
        entry_dict['target'] = ts
        trn = ts[:num_rolling_windows*2*N_output]
        data_train.append(entry_dict)

        for j in range(num_rolling_windows-1, -1, -1):
            entry_dev = dict()

            if j==-1:
                break
            elif j==0:
                seq_dev = ts
            else:
                seq_dev = ts[ : - N_output*j - num_rolling_windows*N_output ]

            entry_dict['target'] = seq_dev
            data_dev.append(entry_dict)
            dev_tsid_map[len(data_dev)-1] = i

        for j in range(num_rolling_windows-1, -1, -1):
            entry_test = dict()

            if j==-1:
                break
            elif j==0:
                seq_test = ts
            else:
                seq_test = ts[ : - N_output*j ]

            entry_dict['target'] = seq_test
            data_test.append(entry_dict)
            test_tsid_map[len(data_dev)-1] = i

    return (
        X, X, X, X, X, X,
        breakpoints, breakpoints, breakpoints,
        data_train, data_dev, data_test,
        dev_tsid_map, test_tsid_map
    )


def create_synthetic_dataset(N, N_input,N_output,sigma):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    N_dev = int(0.2 * N)
    N_test = int(0.2 * N)

    X = []
    breakpoints = []
    for k in range(2*N):
        serie = np.array([ sigma*random.random() for i in range(N_input+N_output)])
        i1 = random.randint(1,10)
        i2 = random.randint(10,18)
        j1 = random.random()
        j2 = random.random()
        interval = abs(i2-i1) + random.randint(-3,3)
        serie[i1:i1+1] += j1
        serie[i2:i2+1] += j2
        serie[i2+interval:] += (j2-j1)
        X.append(serie)
        breakpoints.append(i2+interval)
    X = np.expand_dims(np.stack(X), axis=-1)
    breakpoints = np.array(breakpoints)
    return (
        X[0:N, 0:N_input], X[0:N, N_input:N_input+N_output],
        X[N:N+N_dev, 0:N_input], X[N:N+N_dev, N_input:N_input+N_output],
        X[N+N_dev:N+N_dev+N_test, 0:N_input], X[N+N_dev:N+N_dev+N_test, N_input:N_input+N_output], 
        breakpoints[0:N], breakpoints[N:N+N_dev], breakpoints[N+N_dev:N+N_dev+N_test]
    )

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target, breakpoints):
        super(SyntheticDataset, self).__init__()  
        self.X_input = X_input
        self.X_target = X_target
        self.breakpoints = breakpoints
        
    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        return (self.X_input[idx], self.X_target[idx], self.breakpoints[idx])

