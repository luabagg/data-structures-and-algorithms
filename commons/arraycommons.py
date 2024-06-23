import numpy as np

def populateArray(array_lenght):
    return np.random.choice(range(100), size=array_lenght, replace=True)

def printArray(array):
    print(array)
    print("\n")
