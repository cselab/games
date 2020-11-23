#!/usr/bin/env python3

import PyDSTool as dst
from matplotlib import pyplot as plt
import numpy as np
import time
import pretty_errors
import sys

# Set up model
DSargs = dst.args(name='Off-lattice bargain')

DSargs.pars = { 'beta': 2, 'p2': 0.05  }
DSargs.varspecs = { 'y1': '-p2*y1 + 0.3*exp(beta*y1)                        /(exp(beta*y1)+exp(beta*y2)+exp(beta*y3))',
                    'y2': '-p2*y2 + 0.5*exp(beta*y2)*(exp(beta*y1)+exp(beta*y2))/(exp(beta*y1)+exp(beta*y2)+exp(beta*y3))^2',
                    'y3': '-p2*y3 + 0.7*exp(beta*y3)*(exp(beta*y1))           /(exp(beta*y1)+exp(beta*y2)+exp(beta*y3))^2'}
DSargs.ics = {'y1': 6., 'y2': 1, 'y3': 6.}
# DSargs.ics = {'y1': 4., 'y2': 4, 'y3': 4.}

DSargs.pdomain = {'beta': [0.35, 2]}

testDS = dst.Generator.Vode_ODEsystem(DSargs)

DSargs.tdomain = [0,150]                         # set the range of integration.
ode  = dst.Generator.Vode_ODEsystem(DSargs)     # an instance of the 'Generator' class.


traj = ode.compute('ODE')              # integrate ODE
pts  = traj.sample(dt=0.1)                      # Data for plotting
# plt.plot(pts['t'], pts['y1'])
# plt.plot(pts['t'], pts['y2'])
# plt.plot(pts['t'], pts['y3'])
# plt.show() 
# sys.exit()


# Prepare the system to start close to a steady state
ode.set(pars = {'beta': 2} )       # Lower bound of the control parameter 'beta'
ode.set(ics =  {'y1':pts['y1'][-1],
                'y2':pts['y2'][-1],
                'y3':pts['y3'][-1], } ) # Close to one of the steady states

print('Initial Condition: ',pts['y1'][-1], pts['y2'][-1], pts['y3'][-1])

PC = dst.ContClass(ode)            # Set up continuation class

PCargs = dst.args(name='EQ1', type='EP-C')
PCargs.freepars     = ['beta']              
PCargs.MaxNumPoints = 300
PCargs.MaxStepSize  = 1e-1
PCargs.StepSize     = 1e-2
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen    = True
PCargs.StopAtPoints = 'B'
PCargs.verbosity = 2
PC.newCurve(PCargs)
PC['EQ1'].backward()
PC['EQ1'].info()

PCargs.name='EQ2'
PCargs.type='EP-C'
PCargs.freepars  = ['beta']              
PCargs.initpoint = 'EQ1:P2'
PC.newCurve(PCargs)
PC['EQ2'].forward()
PC['EQ2'].info()

PCargs.name='EQ3'
PCargs.type='H-C1'
PCargs.freepars  = ['beta','p2']              
PCargs.initpoint = 'EQ2:H1'
PC.newCurve(PCargs)
PC['EQ3'].forward()
# PC['EQ3'].backward()
PC['EQ3'].info()

PCargs.name='EQ4'
PCargs.type='LP-C'
PCargs.freepars  = ['beta','p2']              
PCargs.initpoint = 'EQ2:LP1'
PC.newCurve(PCargs)
PC['EQ4'].forward()
# PC['EQ4'].backward()
PC['EQ4'].info()


PC.display(('beta','y1'), figure=1)
PC.display(('beta','y2'), figure=2)
PC.display(('beta','y3'), figure=3)

PC.plot.toggleAll('off', bytype='P')
PC.plot.toggleAll('off', bytype='B')

plt.show()







