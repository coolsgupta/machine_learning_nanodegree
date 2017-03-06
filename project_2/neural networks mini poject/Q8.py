import numpy as np

hidden_weights = np.array([[1,1,-5],[3,-4,2]])
output_weights = np.array([2,-1])
inputs = np.array([1,2,3])

hidden_thetas = []

for i in range(0,len(hidden_weights)):
    x = np.dot(inputs,hidden_weights[i].T)
    print (x)
    hidden_thetas.append(x)
output = np.dot(hidden_thetas,output_weights)
print(output)