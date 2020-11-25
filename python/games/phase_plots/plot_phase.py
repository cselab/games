#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

ms = 7
alpha = 0.4
plt.rcParams.update({ 'font.size': 20 })

fname = 'phase_beta_well_mixed_9-0-3'
fname = 'phase_beta_well_mixed_1-10-1'
fname = 'phase_beta_neumann_4-4-4'

x = np.loadtxt(fname + '.txt')

beta = x[:, 0]

m = np.argmax(beta < 1e-6) + 1

fig, ax = plt.subplots(figsize=(13, 10))
ax.clear()

ax.plot(beta[0:m], x[0:m, 1], 'ro', label=r'Low $\leftarrow$', markersize=ms)
ax.plot(beta[0:m], x[0:m, 1], 'r-', markersize=ms, alpha=alpha)
ax.plot(beta[0:m], x[0:m, 2], 'go', label=r'Med $\leftarrow$', markersize=ms)
ax.plot(beta[0:m], x[0:m, 2], 'g-', markersize=ms, alpha=alpha)
ax.plot(beta[0:m], x[0:m, 3], 'bo', label=r'High $\leftarrow$', markersize=ms)
ax.plot(beta[0:m], x[0:m, 3], 'b-', markersize=ms, alpha=alpha)

ax.plot(beta[m:], x[m:, 1], 'rs', label=r'Low $\rightarrow$', markersize=ms, alpha=alpha)
ax.plot(beta[m:], x[m:, 1], 'r--', markersize=ms, markerfacecolor='none', alpha=alpha)
ax.plot(beta[m:], x[m:, 2], 'gs', label=r'Med $\rightarrow$', markersize=ms, alpha=alpha)
ax.plot(beta[m:], x[m:, 2], 'g--', markersize=ms, markerfacecolor='none', alpha=alpha)
ax.plot(beta[m:], x[m:, 3], 'bs', label=r'High $\rightarrow$', markersize=ms, alpha=alpha)
ax.plot(beta[m:], x[m:, 3], 'b--', markersize=ms, markerfacecolor='none', alpha=alpha)

ax.set_xlabel(r'$\beta$')
ax.set_ylabel('percentages of actions ')

plt.legend(prop={ 'size': 15 })

plt.savefig(fname + '.pdf')

plt.show()
