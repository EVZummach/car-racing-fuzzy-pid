import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from fuzzy import acceleration_fuzzy, breaking_fuzzy

a = np.arange(-50, 50, 1)
v = np.arange(0, 7, 0.5)

breaking = []
for angle in a:
    for vel in v:
        #print(f'Angle:{angle}\tVelocity:{vel}')
        b = breaking_fuzzy(angle, vel)
        if vel != 6.5:
            breaking.append([angle, breaking])


plt.figure(figsize=(10, 5))
plt.plot(np.array(breaking[:,0]), np.array(breaking[:,1]))
plt.show()
