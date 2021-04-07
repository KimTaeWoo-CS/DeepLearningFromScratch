import numpy as np

'''def cross_entropy_error(y, t):
    delta = 1e-7 # np.log 함수에 0을 입력하면 마아니스 무한대를 뜻하기 때문에 아주 작은 값을 더한다.
    return -np.sum(t * np.log(y + delta))'''

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-7)) / batch_size

def cross_entropy_error_not_onehot(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange[batch_size], t] + 1e-7)) / batch_size
