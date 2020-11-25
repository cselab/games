#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import pretty_errors


def J_ode(Y, t, beta=1, gamma=0.05):
    y1, y2, y3 = Y
    y1dot = -gamma * y1 + 0.3 * np.exp(beta * y1) / (np.exp(beta * y1) + np.exp(beta * y2) + np.exp(beta * y3))
    y2dot = -gamma * y2 + 0.5 * np.exp(beta * y2) * (np.exp(beta * y1) + np.exp(beta * y2)) / (
        np.exp(beta * y1) + np.exp(beta * y2) + np.exp(beta * y3)) ** 2
    y3dot = -gamma * y3 + 0.7 * np.exp(beta * y3) * (np.exp(
        beta * y1)) / (np.exp(beta * y1) + np.exp(beta * y2) + np.exp(beta * y3)) ** 2
    return y1dot, y2dot, y3dot


eps = 0.5
x1 = np.linspace(1, 9, 5)
x2 = np.linspace(1, 9, 5)

Y1, Y2 = np.meshgrid(x1, x2)

t = 0

u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

plt.rcParams.update({ 'font.size': 22 })
fig, ax = plt.subplots(figsize=(13, 10))

plt.xlabel(r'$J_1$')
plt.ylabel(r'$J_3$')
plt.xlim([ 0, x1[-1] + 1 ])
plt.ylim([ 0, x2[-1] + 1 ])

for i in range(NI):
    for j in range(NJ):
        y1 = Y1[i, j]
        y2 = Y2[i, j]

        y0 = [ y1, 0.5, y2 ]

        beta = 0.58
        yprime = J_ode(y0, t, beta=beta)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

        tspan = np.linspace(0, 800, 400)
        f = lambda y, t: J_ode(y, t, beta=beta)
        ys = odeint(f, y0, tspan, hmax=1e-1)
        plt.plot(ys[:, 0], ys[:, 2], 'r-', alpha=0.5)  # path
        plt.plot([ys[0, 0]], [ys[0, 2]], 'ko')  # start
        plt.plot([ys[-1, 0]], [ys[-1, 2]], 'rs')  # end
        plt.pause(0.001)
        plt.show(block=False)

# Q = plt.quiver(Y1, Y2, u, v, units='width', color='r', headwidth=2)

plt.savefig('phase-portrait.pdf')
plt.show()
