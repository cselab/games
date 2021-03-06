#!/usr/bin/env python3
'''
Reproduces the bifurcation diagrams from "Persistence of discrimination: Revisiting Axtell, Epstein and Young".
'''
import PyDSTool as dst
from matplotlib import pyplot as plt
import numpy as np
import time
import pretty_errors
import sys

plt.rcParams.update({ 'font.size': 16 })
fig, ax = plt.subplots(figsize=(13, 10))

# Set up model
DSargs = dst.args(name='Off-lattice bargain')

DSargs.pars = { 'beta': 2, 'p2': 0.1 }
DSargs.varspecs = {
    'y1': '-p2*y1 + 0.3*exp(beta*y1)                            /(exp(beta*y1)+exp(beta*y2)+exp(beta*y3))',
    'y2': '-p2*y2 + 0.5*exp(beta*y2)*(exp(beta*y1)+exp(beta*y2))/(exp(beta*y1)+exp(beta*y2)+exp(beta*y3))^2',
    'y3': '-p2*y3 + 0.7*exp(beta*y3)*(exp(beta*y1))             /(exp(beta*y1)+exp(beta*y2)+exp(beta*y3))^2'
}

# J0 = [2.,2.,2.,]; fname='phase_a'
# J0 = [ 1., 0.5, 1. ]; fname='phase_b'
# J0 = [ 8, 0.5, 0.5 ]; fname='phase_c'
J0 = [ 2., 0., 2. ]
fname = 'phase_d'

DSargs.ics = { 'y1': J0[0], 'y2': J0[1], 'y3': J0[2] }

DSargs.pdomain = { 'beta': [ 0.25, 2 ] }

testDS = dst.Generator.Vode_ODEsystem(DSargs)

DSargs.tdomain = [ 0, 150 ]
ode = dst.Generator.Vode_ODEsystem(DSargs)

traj = ode.compute('ODE')
pts = traj.sample(dt=0.1)
# plt.plot(pts['t'], pts['y1'])
# plt.plot(pts['t'], pts['y2'])
# plt.plot(pts['t'], pts['y3'])
# plt.show()
# sys.exit()
#

# Prepare the system to start close to a steady state
ode.set(pars={ 'beta': 2 })  # Lower bound of the control parameter 'beta'
ode.set(ics={
    'y1': pts['y1'][-1],
    'y2': pts['y2'][-1],
    'y3': pts['y3'][-1],
})  # Close to one of the steady states

print('Initial Condition: ', pts['y1'][-1], pts['y2'][-1], pts['y3'][-1])

PC = dst.ContClass(ode)  # Set up continuation class

PCargs = dst.args(name='EQ1', type='EP-C')
PCargs.freepars = ['beta']
PCargs.MaxNumPoints = 300
PCargs.MaxStepSize = 1e-1
PCargs.StepSize = 1e-2
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True
PCargs.StopAtPoints = 'B'
PCargs.verbosity = 0
PC.newCurve(PCargs)
PC['EQ1'].forward()
PC['EQ1'].backward()
PC['EQ1'].info()

# PCargs.name='EQ2'
# PCargs.type='EP-C'
# PCargs.freepars  = ['beta']
# PCargs.initpoint = 'EQ1:P2'
# PC.newCurve(PCargs)
# PC['EQ2'].forward()
# PC['EQ2'].info()

if PC['EQ1'].getSpecialPoint('HP1'):
    PCargs.name = 'EQ3'
    PCargs.type = 'H-C1'
    PCargs.freepars = [ 'beta', 'p2']
    PCargs.initpoint = 'EQ2:H1'
    PC.newCurve(PCargs)
    PC['EQ3'].forward()
    PC['EQ3'].backward()
    PC['EQ3'].info()

if PC['EQ1'].getSpecialPoint('LP1'):
    PCargs.name = 'EQ4'
    PCargs.type = 'LP-C'
    PCargs.freepars = [ 'beta', 'p2']
    PCargs.initpoint = 'EQ1:LP1'
    PC.newCurve(PCargs)
    PC['EQ4'].forward()
    # PC['EQ4'].backward()
    PC['EQ4'].info()

PC.display(('beta', 'y1'), linestyle='-', linewidth=3, label=r'$J_1$', figure=1)
PC.display(('beta', 'y2'), linestyle='--', linewidth=3, label=r'$J_2$', figure=1)
PC.display(('beta', 'y3'), linestyle='-.', linewidth=3, label=r'$J_3$', figure=1)

PC.plot.toggleAll('off', bytype='P')
PC.plot.toggleAll('off', bytype='B')

plt.grid(True)
plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel(r'$J$')

plt.title(rf'$J_1={J0[0]}$   -   $J_2={J0[1]}$   -   $J_3={J0[2]}$')
plt.savefig(fname + '.pdf', dpi=150)

plt.show()
