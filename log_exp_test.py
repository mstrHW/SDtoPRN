import numpy as np
import matplotlib.pyplot as plt

x_init = np.arange(start=-5, stop=5, step=0.001)
x = x_init/5 + 2
x_log = np.log(x)
x_exp = np.exp(x)

plt.plot(x, x_init, label='x')
plt.plot(x, x_log, label='log')
# plt.show()
x_log_exp = np.exp(x_log)
x_rec = (x_log_exp - 2) * 5
# plt.plot(x, x_exp, label='exp')
plt.plot(x, x_log_exp, label='log_exp')
plt.plot(x, x_rec, '--', label='x_rec')
plt.legend()

# answer =
answer = np.exp([0, 2])
print(answer)
# plt.show()
