import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2 * np.pi*t)
s2 = np.sin(4 * np.pi * t)

plt.figure(1)
plt.plot(t, s1)
plt.show()

plt.figure(2)
plt.plot(t, s2)
plt.show()
