'''
Author: Isaac Drachman
Date:   04/22/2016

For:
    Fluid Dynamics, NYU Spring 2016, Prof. MacFadyen. Implements Euler code
    described in handout for final project.

Dependencies: PIL, numpy, matplotlib
Other: images2gif is a dependency whose source is included with this project.

Usage instructions:
    You may run $ python grid.py
    View comments in main section to see how parameters may be altered.
'''

import numpy as np
import PIL
import matplotlib
import sys
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from images2gif import writeGif

'''
These functions along with accompanying file 'images2gif' handle
the plotting to gif functionality to animate the simulation.
'''
def fig2data(fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    return np.roll(buf, 3, axis=2)
def fig2img(fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return PIL.Image.frombytes('RGBA', (w, h), buf.tostring()).convert(mode='RGB')


# Gamma constant for gas.
GAMMA = 1.7

# Main simulation class.
class Grid1D:
    '''
    The grid class tracks U, the matrix of
    conserved quantities at each point along
    the x-axis. Each column of U represents
    a point and stores mass, momentum, and energy
    in each row.
    '''
    def __init__(self, points):
        self.points = points
        self.dx = 1. / points
        self.U = np.zeros((3, points))

    # Compute fields from U (conserved quantities).
    # Compute density (rho) at each point.
    def density(self):
        return self.U[0, :]

    # Compute velocity (u) at each point.
    def velocity(self):
        return self.U[1, :] / self.U[0, :]

    # Compute total energy (E) at each point.
    def total_energy(self):
        return self.U[2, :]

    # Compute internal energy (e) at each point.
    def internal_energy(self):
        rho = self.density()
        v   = self.velocity()
        E   = self.total_energy()
        return (E - 0.5 * rho * v**2) / rho

    # Compute pressure (p) at each point.
    def pressure(self):
        return (GAMMA - 1.)*self.density()*self.internal_energy()

    '''
    Compute flux of conserved quantities at
    each point. Same convention as U, namely
    each column is point along x-axis with
    mass, momentum, and energy in each row.
    '''
    def flux(self):
        rho = self.density()
        v   = self.velocity()
        E   = self.total_energy()
        P   = self.pressure()

        F = np.zeros((3, self.points))
        F[0, :] = rho * v
        F[1, :] = rho * v**2 + P
        F[2, :] = (E + P)*v
        return F

    # Compute speed of sound from pressure and density.
    def speed_of_sound(self):
        return np.sqrt(GAMMA * self.pressure() / self.density())

    # Compute eigenvalues lambda+ and lambda- as described in the handout.
    def eigenvals(self):
        return self.velocity() + self.speed_of_sound(), self.velocity() - self.speed_of_sound()

    # HLL method for computing flux at half-points on grid.
    def HLL(self, lambda_minus_L, lambda_minus_R, lambda_plus_L, lambda_plus_R,
                  flux_L, flux_R, conserved_L, conserved_R):
        alpha_plus = max(0., lambda_plus_L, lambda_plus_R)
        alpha_minus = max(0., -lambda_minus_L, -lambda_minus_R)
        # print(self.dx/max(alpha_plus, alpha_minus))
        return (alpha_plus*flux_L + alpha_minus*flux_R \
                - alpha_plus*alpha_minus*(conserved_R - conserved_L)) \
                / (alpha_plus + alpha_minus)

    # Advance U one time step.
    def step(self, dt):
        lambda_plus, lambda_minus = self.eigenvals()
        F = self.flux()
        newU = self.U.copy()
        for idx in range(0, self.U.shape[1]):
            left_minus = idx-1; left_plus = idx
            right_minus = idx; right_plus = idx+1
            if idx == 0:
                left_minus = idx
            elif idx == self.U.shape[1] - 1:
                right_plus = idx
            F_minus = self.HLL(lambda_minus[left_minus], lambda_minus[right_minus],
                               lambda_plus[left_minus], lambda_plus[right_minus],
                               F[:, left_minus], F[:, right_minus],
                               self.U[:, left_minus], self.U[:, right_minus])
            F_plus = self.HLL(lambda_minus[left_plus], lambda_minus[right_plus],
                              lambda_plus[left_plus], lambda_plus[right_plus],
                              F[:, left_plus], F[:, right_plus],
                              self.U[:, left_plus], self.U[:, right_plus])
            newU[:, idx] += - dt/self.dx * (F_plus - F_minus)
        self.U = newU

# Subclass of grid for this particular example.
class ShockTube(Grid1D):
    # L and R are tuples of form (pressure, density, velocity).
    def __init__(self, points, L, R):
        Grid1D.__init__(self, points)
        midpoint = int(np.floor(self.points / 2))

        # Set densities.
        self.U[0, :midpoint] = np.repeat(L[1], midpoint)
        self.U[0, midpoint:] = np.repeat(R[1], midpoint)

        # Set momentums.
        self.U[1, :midpoint] = np.repeat(L[1]*L[2], midpoint)
        self.U[1, midpoint:] = np.repeat(R[1]*R[2], midpoint)

        # Compute internal and total energy from other params.
        e_L = L[0] / ((GAMMA - 1.) * L[1])
        e_R = R[0] / ((GAMMA - 1.) * R[1])

        E_L = L[1] * (e_L + 0.5*L[2]**2)
        E_R = R[1] * (e_R + 0.5*R[2]**2)

        # Set total energy.
        self.U[2, :midpoint] = np.repeat(E_L, midpoint)
        self.U[2, midpoint:] = np.repeat(E_R, midpoint)

if __name__ == "__main__":
    flag = "easy"
    if len(sys.argv) == 2:
        flag = sys.argv[1]

    # Do some setup. L/R tube states from handout.
    if flag == "easy":
        # This is original "easier" shock tube.
        L = (1.0, 1.0, 0.0)
        R = (0.125, 0.1, 0.0)
    elif flag == "hard":
        # This is the second "harder" shock tube.
        L = (100.0, 10.0, 0.0)
        R = (1.0, 1.0, 0.0)

    '''
    Time delta is picked to be sufficiently small.
    Length of simulated time is (dt)*(total_steps)
    A frame is plotted every (dt)*(plot_interval) of simulated time
    Total time of gif is (total_steps)/(plot_interval)/(gif_fps)
    Slowdown of simulated time for animation is (total simulated time)(total time of gif)

    dt = 5e-4
    total_steps = 1250
    plot_interval = 5
    gif_fps = 20

    For the above example values we have:
      0.625 seconds are simulated
      frame plotted every 0.0025 seconds of simulated time
      total time of gif is 12.5 seconds
      slowdown is 20x
    '''
    dt = 5e-4
    total_steps = 1250
    plot_interval = 5
    gif_fps = 20

    if flag == "hard":
        # For the harder shocktube we want to scale up time resolution
        # and shorten length of simulation.
        dt /= 10.
        total_steps *= 4
        plot_interval *= 4

    # Determine number of points on grid.
    points = 500

    # Initialize tube and setup figure to plot.
    tube = ShockTube(points, L, R)
    figure = plt.figure(figsize=(12,15))

    # We will plot 3 graphs on this figure.
    # At each frame we plot pressure, density, and velocity along the tube.
    pressure_plot = figure.add_subplot(311)
    density_plot = figure.add_subplot(312)
    velocity_plot = figure.add_subplot(313)

    # Some plot settings. We setup a list to hold the frames.
    pressure_plot.hold(False)
    density_plot.hold(False)
    velocity_plot.hold(False)
    images = []

    # For each time step.
    for step_num in range(total_steps):
        # Run our numerical PDE solver one time step.
        tube.step(dt)

        # If we need to plot this step.
        if step_num % plot_interval == 0:
            # Plot each field. Label with timestamp. Set axes all (0.0, 1.0).
            pressure_plot.plot(range(tube.points), tube.pressure(), lw=2, label='pressure @ %0.4f seconds' % (step_num*dt))
            pressure_plot.legend(loc='upper left')
            if flag == "easy":
                pressure_plot.set_ylim(0.0, 1.0)
            elif flag == "hard":
                pressure_plot.set_ylim(0.0, 100.0)

            density_plot.plot(range(tube.points), tube.density(), lw=2, label='density @ %0.4f seconds' % (step_num*dt))
            density_plot.legend(loc='upper left')
            if flag == "easy":
                density_plot.set_ylim(0.0, 1.0)
            elif flag == "hard":
                density_plot.set_ylim(0.0, 10.0)

            velocity_plot.plot(range(tube.points), tube.velocity(), lw=2, label='velocity @ %0.4f seconds' % (step_num*dt))
            velocity_plot.legend(loc='upper left')
            if flag == "easy":
                velocity_plot.set_ylim(0.0, 1.0)
            elif flag == "hard":
                velocity_plot.set_ylim(0.0, 5.0)

            # Add to images.
            images.append(fig2img(figure))
    writeGif('shocktube-%s.gif' % flag, images, duration=1./gif_fps, dither=0)
    plt.close(figure)
