import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7 # np.log 함수에 0을 입력하면 마아니스 무한대를 뜻하기 때문에 아주 작은 값을 더한다.
    return -np.sum(t * np.log(y + delta))

