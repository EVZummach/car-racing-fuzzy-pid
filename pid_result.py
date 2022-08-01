import os
import numpy as np
import matplotlib.pyplot as plt

results = os.path.join(os.getcwd(),'results')

files = os.listdir(results)
print(files)

for file in files:
    filename = os.path.join(results, file)
    content = np.genfromtxt(filename)
    x = np.arange(0, len(content), 1)
    print(content.shape)
    plt.figure()
    plt.plot(x, content[:,0], label='Error')
    #plt.plot(x, content[:,1], label='Angle')
    plt.legend()
    plt.title(f'Error and Steering Signal: {file}')
    plt.show()
