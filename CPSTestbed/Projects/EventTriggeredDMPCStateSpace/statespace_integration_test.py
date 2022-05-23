import sys
import numpy as np
import diversipy.hycusampling as hycusampling
import matplotlib.pyplot as plt


sys.path.append("../../")
sys.path.append("../../../gym-pybullet-drones")

import CPSTestbed.Projects.EventTriggeredDMPCStateSpace.trajectory_generation.statespace_model as statespace
from CPSTestbed.Projects.EventTriggeredAntiCollision.trajectory_generation.interpolation import TrajectoryCoefficients

if __name__ == "__main__":



    fundamentalF = 2
    t = np.linspace(0, 10, int(10 / fundamentalF) + 1)

    u = np.tile(t[:-1], (3, 1)).T
    u = TrajectoryCoefficients(u, True, u)

    x_k = np.zeros((9,))

    sim_time = np.linspace(0, 10, int(10 / 0.1) + 1)

    derivative = 0

    statespace_model = statespace.TripleIntegrator(t, 3, 9, fundamentalF)

    x = statespace_model.interpolate(9.5, u, derivative_order=0, x0=x_k, integration_start=9.1)

    Psi = statespace_model.get_state_trajectory_vector_matrix(sim_time, derivative_order=0)
    Lambda = statespace_model.get_input_trajectory_vector_matrix(sim_time, derivative_order=derivative)

    x_vec_mat = Psi @ x_k + Lambda @ u.coefficients.reshape((len(t[:-1]) * 3,))
    x_vec_mat = np.reshape(x_vec_mat, (len(sim_time), 3))
    x_vec = statespace_model.interpolate_vector([8], u, derivative_order=derivative, x0=x_k)
    print('Pause')

