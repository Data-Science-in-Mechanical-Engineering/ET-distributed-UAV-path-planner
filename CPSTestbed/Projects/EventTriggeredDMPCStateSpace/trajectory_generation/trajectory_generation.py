"""contains code for trajectory generation

includes the computing node agent
"""
import math
from dataclasses import dataclass
from CPSTestbed.Projects.EventTriggeredAntiCollision.trajectory_generation.interpolation import \
    PiecewiseBernsteinPolynomial
from CPSTestbed.Projects.EventTriggeredAntiCollision.trajectory_generation.interpolation import TrajectoryCoefficients
from CPSTestbed.Projects.EventTriggeredAntiCollision.trajectory_generation.interpolation import Interpolation
import numpy as np
import qpsolvers as qp
from cvxopt import matrix, solvers
import copy
import matplotlib.pyplot as plt
import time
from scipy.io import savemat

import os


@dataclass
class TrajectoryGeneratorOptions:
    """this class represents tjhe options for the trajectory generator

    Parameters
    ---------
        order_polynom: int
            order of interpolation polynom
        objective_function_sample_points: np.array
            objective function is integrated. However for this trajectory generator it is approximated by summation
            over objective_function_sample_points.
        collision_constraint_sample_points: np.array
            collision constraint is evaluated at this points
        state_constraint_sample_points: np.array
            position/speed/acc/jerk constrint is evaluated at this points
        prediction_horizon: int
            how many time intervals to calculate (i.e. trajectory will be delta_t_statespace*prediction_horizon long)
        weight_state_difference: np.array, shape(3, 3)
            positive definite weight matrix for difference between trajectory and target state
        weight_state_derivative_1_difference: np.array, shape(3, 3)
            positive definite weight matrix for first order derivative (speed)
        weight_state_derivative_2_difference: np.array, shape(3, 3)
            positive definite weight matrix for second order derivative (acceleration)
        weight_state_derivative_3_difference: np.array, shape(3, 3)
            positive definite weight matrix for second order derivative (jerk)
        objective_function_sampling_delta_t: float
            time step for sampling the integration should be divide with delta_t_statespace by an integer
        max_position: np.array, shape(3,)
            maximum position the drone can reach
        max_speed: np.array, shape(3, )
            maximum speed the drone can reach
        max_acceleration: np.array, shape(3, )
            maximum accelecration the drone can reach
        max_jerk: np.array, shape(3, )
            maximum jerk the drone can reach
        min_position: np.array, shape(3,)
            minimum position the drone can reach
        r_min: float
            minimum distance between planned trajectory and other trajectories
        optimization_variable_sample_time: float
            sample time of the optimization variable, Communication sample time has to be a integer multiple

    """
    objective_function_sample_points: any
    collision_constraint_sample_points: any
    state_constraint_sample_points: any
    weight_state_difference: any
    weight_state_derivative_1_difference: any
    weight_state_derivative_2_difference: any
    weight_state_derivative_3_difference: any
    max_position: any
    max_speed: any
    max_acceleration: any
    max_jerk: any
    min_position: any
    r_min: float
    optimization_variable_sample_time: float
    num_drones: int
    skewed_plane_BVC: bool
    use_qpsolvers: bool
    downwash_scaling_factor: float
    use_soft_constraints: bool
    guarantee_anti_collision: bool
    soft_constraint_max: float
    weight_soft_constraint: float


class TrajectoryGenerator:
    """generator for the trajectories
    Builds the optimization problem and solves it

    """

    def __init__(self, options,
                 trajectory_interpolation: Interpolation):
        """constructor
        Parameters
        ----------
            options: TrajectoryGeneratorOptions
                options fot the generator
            trajectory_interpolation:
                interpolation for trajectory
        """
        self.__options = options
        self.__trajectory_interpolation = trajectory_interpolation
        self.__breakpoints = trajectory_interpolation.breakpoints
        self.__objective_function_sample_points = options.objective_function_sample_points
        self.__num_objective_function_sample_points = len(self.__objective_function_sample_points)
        self.__collision_constraint_sample_points = options.collision_constraint_sample_points
        self.__num_collision_constraint_sample_points = len(self.__collision_constraint_sample_points)
        self.__state_constraint_sample_points = options.state_constraint_sample_points
        self.__num_state_constraint_sample_points = len(self.__state_constraint_sample_points)
        self.__dim = self.__trajectory_interpolation.dimension

        self.__use_soft_constraints = int(options.use_soft_constraints)
        self.__soft_constraint_max = options.soft_constraint_max
        self.__guarantee_anti_collision = int(options.guarantee_anti_collision)

        self.__prediction_horizon = trajectory_interpolation.num_intervals

        self.__upper_boundaries = [options.max_position, options.max_speed, options.max_acceleration, options.max_jerk]
        self.__lower_boundaries = [options.min_position, -options.max_speed, -options.max_acceleration,
                                   -options.max_jerk]

        self.__num_anti_collision_constraints = (options.num_drones - 1) * self.__num_collision_constraint_sample_points
        self.__num_optimization_variables = trajectory_interpolation.num_optimization_variables + \
            self.__use_soft_constraints*self.__num_anti_collision_constraints
        # initialize all matrixes with zeros (allocate their memory)
        # objective function = x' Q x + 2x' P
        self.__Q = np.zeros((self.__num_optimization_variables, self.__num_optimization_variables))

        self.__P = np.zeros((self.__dim * len(self.__objective_function_sample_points),
                             self.__num_optimization_variables))

        # initialize Aeq, end constraints: speed, acceleration and jerk = 0 for feasibility
        self.__A_eq = np.zeros((trajectory_interpolation.num_equality_constraints + self.__dim * 3,
                                self.__num_optimization_variables))

        self.__b_eq = np.zeros((len(self.__A_eq),))

        # build inequality constraint
        num_unequality_constraints = self.__num_anti_collision_constraints*(1+2*self.__use_soft_constraints) + \
                                     3 * self.__dim * 2 * self.__num_state_constraint_sample_points + \
                                     2 * self.__trajectory_interpolation.num_optimization_variables  # 2 for position, speed, acc TODO: jerk

        self.__A_uneq = np.zeros((num_unequality_constraints,
                                  self.__num_optimization_variables))
        self.__b_uneq = np.ones((num_unequality_constraints,))

        self.__coefficients_to_trajectory_matrix_objective = None
        self.__coefficients_to_trajectory_derivative_1_matrix_objective = None
        self.__coefficients_to_trajectory_derivative_2_matrix_objective = None
        self.__coefficients_to_trajectory_derivative_3_matrix = None

        self.initialize_optimization()

    def initialize_optimization(self):
        """initializes optimization by building matrixes that are constant for every optimization instance"""

        # build matrix that allow to calculate position/speed at objective sample points out of the coefficients
        self.__coefficients_to_trajectory_matrix_objective = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__objective_function_sample_points, derivative_order=0)
        self.__coefficients_to_trajectory_derivative_1_matrix_objective = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__objective_function_sample_points, derivative_order=1)
        self.__coefficients_to_trajectory_derivative_2_matrix_objective = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__objective_function_sample_points, derivative_order=2)

        # build matrix that allow to calculate position/speed at state constraints sample points out of the coefficients
        self.__coefficients_to_trajectory_matrix_state = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=0)
        self.__coefficients_to_trajectory_derivative_1_matrix_state = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=1)
        self.__coefficients_to_trajectory_derivative_2_matrix_state = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=2)

        # build matrix that allow to calculate position/speed at collision constraints sample points out of the
        # coefficients
        self.__coefficients_to_trajectory_matrix_collision = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__collision_constraint_sample_points, derivative_order=0)

        # build matrix that allow to calculate position/speed at objective sample points out of the initial state
        self.__state_to_trajectory_matrix_state = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=0)
        self.__state_to_trajectory_derivative_1_matrix_state = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=1)
        self.__state_to_trajectory_derivative_2_matrix_state = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=2)

        # build matrix that allow to calculate position/speed at objective sample points out of the initial state
        self.__state_to_trajectory_matrix_objective = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__objective_function_sample_points, derivative_order=0)

        # build matrix that allow to calculate position/speed at objective sample points out of the initial state
        self.__state_to_trajectory_matrix_collision = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__collision_constraint_sample_points, derivative_order=0)

        # build objective function matrizes
        Q = np.zeros((self.__trajectory_interpolation.num_optimization_variables,
                      self.__trajectory_interpolation.num_optimization_variables))
        Q_tilde = np.kron(np.eye(self.__num_objective_function_sample_points, dtype=int),
                          self.__options.weight_state_difference)
        Q += self.__coefficients_to_trajectory_matrix_objective.T @ Q_tilde @ \
                    self.__coefficients_to_trajectory_matrix_objective

        Q_tilde = np.kron(np.eye(self.__num_objective_function_sample_points, dtype=int),
                          self.__options.weight_state_derivative_1_difference)
        Q += self.__coefficients_to_trajectory_derivative_1_matrix_objective.T @ Q_tilde @ \
                    self.__coefficients_to_trajectory_derivative_1_matrix_objective

        Q_tilde = np.kron(np.eye(self.__num_objective_function_sample_points, dtype=int),
                          self.__options.weight_state_derivative_2_difference)
        Q += self.__coefficients_to_trajectory_derivative_2_matrix_objective.T @ Q_tilde @ \
                    self.__coefficients_to_trajectory_derivative_2_matrix_objective

        Q_tilde = np.kron(np.eye(self.__prediction_horizon), self.__options.weight_state_derivative_3_difference)
        Q += Q_tilde

        self.__Q[0:self.__trajectory_interpolation.num_optimization_variables,
                 0:self.__trajectory_interpolation.num_optimization_variables] = Q

        if self.__use_soft_constraints == 1:
            self.__Q[self.__trajectory_interpolation.num_optimization_variables:,
                     self.__trajectory_interpolation.num_optimization_variables:] = \
                np.kron(np.eye(self.__num_optimization_variables-self.__trajectory_interpolation.num_optimization_variables),
                        self.__options.weight_soft_constraint)

        P_tilde = np.kron(np.eye(self.__num_objective_function_sample_points, dtype=int),
                          self.__options.weight_state_difference)
        self.__P[:, 0:self.__trajectory_interpolation.num_optimization_variables] = P_tilde @ \
            self.__coefficients_to_trajectory_matrix_objective


        # build equality matrizes
        # first from trajectory, second start/end constraints
        a = self.__trajectory_interpolation.num_equality_constraints  # just to make the following shorter
        self.__A_eq[0:a, 0:self.__trajectory_interpolation.num_optimization_variables], self.__b_eq[0:a] = \
            self.__trajectory_interpolation.equality_constraint

        """
        for i in range(0, 3):
            self.__A_eq[a + self.__dim * i:a + self.__dim * (i + 1)] = \
                self.__trajectory_interpolation.get_trajectory_vector_matrix([0], derivative_order=i)
        """
        for i in range(0, 2):
            self.__A_eq[a + self.__dim * i: a + self.__dim * (i + 1), 0:self.__trajectory_interpolation.num_optimization_variables] = \
                self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
                    [self.__trajectory_interpolation.breakpoints[-1]], derivative_order=i + 1)

        self.__A_eq[a + 2 * self.__dim:a + 3 * self.__dim,
            self.__trajectory_interpolation.num_optimization_variables-self.__dim:self.__trajectory_interpolation.num_optimization_variables] = \
            np.eye(self.__dim)

        # Inequality Constraints
        # position/speed/acc/jerk limit
        for i in range(0, 3):
            # x < ub
            self.__A_uneq[
            i * 2 * self.__dim * self.__num_state_constraint_sample_points:
            i * 2 * self.__dim * self.__num_state_constraint_sample_points +
            self.__dim * self.__num_state_constraint_sample_points, 0:self.__trajectory_interpolation.num_optimization_variables] = \
                self.input_to_trajectory_vector_matrix_state_wrapper(derivative=i)

            # lb < x <=> -x < -lb
            self.__A_uneq[
            i * 2 * self.__dim * self.__num_state_constraint_sample_points +
            self.__dim * self.__num_state_constraint_sample_points:
            (i + 1) * 2 * self.__dim * self.__num_state_constraint_sample_points, 0:self.__trajectory_interpolation.num_optimization_variables] = \
                -self.input_to_trajectory_vector_matrix_state_wrapper(derivative=i)

        # limit every optimization variable instead of only at the state constraint sample points
        a = self.__num_anti_collision_constraints
        b = self.__num_state_constraint_sample_points
        c = self.__trajectory_interpolation.num_optimization_variables
        n = 3 * 2 * self.__dim
        # Ax < b -> x < ub
        self.__A_uneq[n * b:n * b + c, 0:self.__trajectory_interpolation.num_optimization_variables] = np.eye(c)
        self.__b_uneq[n * b:n * b + c] = np.tile(self.__upper_boundaries[3], self.__prediction_horizon)
        # lb < x -> -x < -lb
        self.__A_uneq[n * b + c:n * b + 2 * c,0:self.__trajectory_interpolation.num_optimization_variables] = -np.eye(c)
        self.__b_uneq[n * b + c:n * b + 2 * c] = - np.tile(self.__lower_boundaries[3],
                                                           self.__prediction_horizon)

        # limit weak constraint factors
        if self.__use_soft_constraints == 1:
            offset = self.__num_anti_collision_constraints + \
                3 * self.__dim * 2 * self.__num_state_constraint_sample_points + \
                2 * self.__trajectory_interpolation.num_optimization_variables
            self.__A_uneq[offset:offset+self.__num_anti_collision_constraints,
                self.__trajectory_interpolation.num_optimization_variables:] = np.eye(self.__num_anti_collision_constraints)
            self.__b_uneq[offset:offset + self.__num_anti_collision_constraints] = \
                np.zeros((self.__num_anti_collision_constraints, ))

            offset += self.__num_anti_collision_constraints
            self.__A_uneq[offset:offset + self.__num_anti_collision_constraints,
            self.__trajectory_interpolation.num_optimization_variables:] = -np.eye(self.__num_anti_collision_constraints)
            self.__b_uneq[offset:offset + self.__num_anti_collision_constraints] = \
                np.ones((self.__num_anti_collision_constraints,))*self.__options.weight_soft_constraint

    def calculate_trajectory(self, current_state,
                             target_position,
                             planned_trajectory,
                             other_trajectories,
                             previous_solution,
                             drone_id,
                             timestep,
                             other_targets,
                             other_ids):
        """
        calculates trajectory
        Parameters
        ----------
        current_state: np.array, shape(9,)
            current state of the agent
        target_position: np.array, shape(3,)
            target of the drone
        planned_trajectory: np.array, shape(., 3)
            trajectory that was planned a timestep ago. Should start with the first trajectory point to optimize.
        other_trajectories: np.array, shape(., ., 3)
            trajectories of other agents/obstacles. First dimension is the obstalce/agent, second the time
            (sampling time delta_t_statespace) and third the position.
            Should start with the first trajectory point to optimize.
        previous_solution: TrajectoryCoefficients
            previous solution of optimization problem (in case the solver fails due to numerical issues)

        Returns
        -------
        optimal_coefficients: TrajectoryCoefficients
            if not valid, then the planned trajectory will be returned
        """
        # TODO check if trajectories have the right dimension
        # build equality constraint right side
        a = self.__trajectory_interpolation.num_equality_constraints
        self.__b_eq[a: a + self.__dim] = - self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            [self.__trajectory_interpolation.breakpoints[-1]], derivative_order=1) @ current_state
        self.__b_eq[
        a + self.__dim: a + 2 * self.__dim] = - self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            [self.__trajectory_interpolation.breakpoints[-1]], derivative_order=2) @ current_state
        self.__b_eq[a + 2 * self.__dim: a + 3 * self.__dim] = np.zeros((self.__dim,))

        # inequality state constraints
        for i in range(0, 3):
            future_state = self.state_to_trajectory_vector_matrix_state_wrapper(derivative=i) @ current_state

            ub = np.tile(self.__upper_boundaries[i], self.__num_state_constraint_sample_points) - future_state
            self.__b_uneq[
            i * 2 * self.__dim * self.__num_state_constraint_sample_points:
            i * 2 * self.__dim * self.__num_state_constraint_sample_points +
            self.__dim * self.__num_state_constraint_sample_points] = ub

            lb = -np.tile(self.__lower_boundaries[i], self.__num_state_constraint_sample_points) + future_state
            self.__b_uneq[i * 2 * self.__dim *
                          self.__num_state_constraint_sample_points + self.__dim *
                          self.__num_state_constraint_sample_points:
                          (i + 1) * 2 * self.__dim *
                          self.__num_state_constraint_sample_points] = lb

        b = 3 * self.__dim * 2 * self.__num_state_constraint_sample_points + \
            2 * self.__trajectory_interpolation.num_optimization_variables

        # apply calculated anti collision constraint to __A_uneq and b_uneq
        for i in range(0, other_trajectories.shape[0]):
            # A, b = build_anti_collision_contraint
            self.__A_uneq[b + i * self.__num_collision_constraint_sample_points:
                          b + (i + 1) * self.__num_collision_constraint_sample_points, :], \
            self.__b_uneq[b + i * self.__num_collision_constraint_sample_points:
                          b + (i + 1) * self.__num_collision_constraint_sample_points] = \
                self.build_anti_collision_constraint(planned_trajectory, other_trajectories[i, :, :], current_state,
                                                     target_position, other_targets[i, :], other_ids[i], drone_id)

        # solve optimization problem
        future_pos = self.__state_to_trajectory_matrix_objective @ current_state
        q = - (np.tile(target_position, self.__num_objective_function_sample_points) - future_pos) @ self.__P
        optimal_coefficients = None

        if self.__options.use_qpsolvers:
            optimal_coefficients = qp.solve_qp(P=self.__Q,
                                               q=q,
                                               G=self.__A_uneq, h=self.__b_uneq,
                                               A=self.__A_eq, b=self.__b_eq)
            # due to numerical issues
            if optimal_coefficients is None:
                print("No Solution Found")
                return TrajectoryCoefficients(None, False, previous_solution)
        else:
            P = matrix(self.__Q)
            q = matrix(q)
            G = matrix(self.__A_uneq)
            h = matrix(self.__b_uneq)
            A = matrix(self.__A_eq)
            b = matrix(self.__b_eq)

            try:
                optimal_coefficients = solvers.qp(P, q, G, h, A, b)
                successful = optimal_coefficients['status'] == 'optimal'
                optimal_coefficients = optimal_coefficients['x']
                optimal_coefficients = np.squeeze(np.asarray(optimal_coefficients))
            except ValueError:
                successful = False

            if not successful:
                print("No Solution Found")
                return TrajectoryCoefficients(None, False, previous_solution)


        # reshaped: optimale Koeffizienten auf die prediction horizons aufteilen
        reshaped = np.reshape(optimal_coefficients[0:self.__trajectory_interpolation.num_optimization_variables],
                              (self.prediction_horizon, self.__dim))

        return_value = TrajectoryCoefficients(copy.deepcopy(reshaped),
                                              True, previous_solution)
        return return_value

    def build_anti_collision_constraint(self, planned_trajectory, other_trajectory, current_state, target_position,
                                        other_target, other_id, drone_id):
        """build anti collision constraint A*coefficients <= b for one trajectory

        Parameters
        ----------
            planned_trajectory: np.array, shape(., 3)
                trajectory that was planned a timestep ago. Should start with the first trajectory point to optimize.
            other_trajectory: np.array, shape(., 3)
                trajectories of an other agent.
                Should start with the first trajectory point to optimize.
            current_state: np.array, shape(3,)
                state of the drone at the beginning of the prediction horizon
            target_position: np.array, shape(3,)
                target position of the drone

        Returns
        -------
            A: np.array, shape(., number optimization parameters)
                anti collision constraint matrix
            b: np.array, shape(.,)
                right side
        """
        min_angle = 20.0 * math.pi / 180  # min angle in radians
        downwash_coefficient = self.__options.downwash_scaling_factor
        downwash_scaling = np.diag([1, 1, 1.0 / float(downwash_coefficient)])
        A = np.zeros((planned_trajectory.shape[0], self.__num_optimization_variables))
        b = np.zeros((planned_trajectory.shape[0],))
        future_pos_from_current_state = self.__state_to_trajectory_matrix_collision @ current_state
        future_pos_from_current_state = np.reshape(future_pos_from_current_state,
                                                   (planned_trajectory.shape[0], self.__dim))
        for i in range(0, planned_trajectory.shape[0]):
            # build normal vector
            relative_trajectory = (planned_trajectory[i] - other_trajectory[i]) @ downwash_scaling
            own_dist_to_target = (target_position - planned_trajectory[i]) @ downwash_scaling
            other_dist_to_target = (other_target - other_trajectory[i]) @ downwash_scaling

            if self.__options.skewed_plane_BVC:
                normal_vector_bvc = self.calc_BVC_normal_vector(relative_trajectory, own_dist_to_target,
                                                            other_dist_to_target, min_angle, own_id=drone_id,
                                                            other_id=other_id)
            else:
                normal_vector_bvc = relative_trajectory

            distance = np.linalg.norm(normal_vector_bvc)
            n_0 = normal_vector_bvc / distance

            if abs(n_0 @ ((planned_trajectory[i] - other_trajectory[
                i]) @ downwash_scaling)) < self.__options.r_min - 0.001:  # -0.001 to ignore numerical errors
                n_0 = relative_trajectory / np.linalg.norm(relative_trajectory)  # use non-skewed plane in case of collision

            # right side
            b[i] = -(self.__options.r_min + self.__use_soft_constraints*self.__options.soft_constraint_max +
                     distance*self.__guarantee_anti_collision) / 2.0 - np.dot(n_0, (other_trajectory[i] -
                                                                           future_pos_from_current_state[
                                                                               i]) @ downwash_scaling)

            # left side
            A[i, 0:self.__trajectory_interpolation.num_optimization_variables] = \
                -n_0 @ downwash_scaling @ self.__coefficients_to_trajectory_matrix_collision[
                i * self.__dim:(i + 1) * self.__dim, :]
            if self.__use_soft_constraints == 1:
                A[i, self.__trajectory_interpolation.num_optimization_variables + i] = 1

        return A, b

    def input_to_trajectory_vector_matrix_state_wrapper(self, derivative):
        """ returns the desired trajectory vector matrix for state constraint sample points"""

        if derivative == 0:
            return self.__coefficients_to_trajectory_matrix_state
        elif derivative == 1:
            return self.__coefficients_to_trajectory_derivative_1_matrix_state
        elif derivative == 2:
            return self.__coefficients_to_trajectory_derivative_2_matrix_state
        else:
            return None

    def state_to_trajectory_vector_matrix_state_wrapper(self, derivative):
        """ returns the desired trajectory vector matrix for state constraint sample points"""

        if derivative == 0:
            return self.__state_to_trajectory_matrix_state
        elif derivative == 1:
            return self.__state_to_trajectory_derivative_1_matrix_state
        elif derivative == 2:
            return self.__state_to_trajectory_derivative_2_matrix_state
        else:
            return None

    def calc_BVC_normal_vector(self, relative_trajectory, own_dist_to_target, other_dist_to_target, min_angle, own_id,
                               other_id):
        """ calculates the normal vector that should be used for the BVC of the anti collision constraint

        Parameters:
            own_position: np.array, shape(3,)
                position of current anti collision sample point of the own planned trajectory
            other_position: np.array, shape(3,)
                position of current anti collision sample point of the other drone's planned trajectory
            own_target: np.array, shape(3,)
                own target position
            other_target: np.array, shape(3,)
                other drone's target position
            """
        own_dist = np.linalg.norm(own_dist_to_target)
        other_dist = np.linalg.norm(other_dist_to_target)
        tol = 0.1
        if (own_dist - other_dist) > tol:
            bvc_normal_vector = -self.rotate_plane_outside_cone(own_dist_to_target, min_angle, -relative_trajectory,
                                                                positive_rotation=True)
        elif (own_dist - other_dist) < -tol:
            bvc_normal_vector = self.rotate_plane_outside_cone(other_dist_to_target, min_angle, relative_trajectory,
                                                               positive_rotation=True)
        else:
            if own_id > other_id:
                bvc_normal_vector = -self.rotate_plane_outside_cone(own_dist_to_target, min_angle, -relative_trajectory,
                                                                    positive_rotation=True)
            else:
                bvc_normal_vector = self.rotate_plane_outside_cone(other_dist_to_target, min_angle, relative_trajectory,
                                                                   positive_rotation=True)
        return bvc_normal_vector

    def rotate_plane_outside_cone(self, cone_axis, cone_angle, plane_normal_vector, positive_rotation=True):
        """ Rotates the normal vector of a plane so that it is outside of a cone defined by the cone_axis and the
        cone_angle

        Parameters:
            cone_axis: np.array, shape(3,)
                central axis of the cone
            cone_angle: float
                opening angle of the cone's tip
            plane_normal_vector: np.array, shape(3,)
                normal vector of the plane that should be rotated
            positive_rotation: bool
                choose, whether the vector should be rotated in the mathematically positive (right hand) direction
            """

        angle = self.angle_between(cone_axis, plane_normal_vector)
        if angle == 0:
            plane_normal_vector += np.array([0, 0, 0.01])  # if vectors are parallel, rotate plane vector up
            angle = self.angle_between(cone_axis, plane_normal_vector)
        vn = np.cross(cone_axis, plane_normal_vector)

        angle_difference = max((cone_angle - angle, 0))
        if not positive_rotation:
            angle_difference *= -1
        plane_normal_vector = self.rotate_vector(vn, angle_difference, plane_normal_vector)
        return plane_normal_vector

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return np.divide(vector, 1.0 * np.linalg.norm(vector))

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotate_vector(self, rot_axis, theta, vector):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        if theta != 0:
            rot_axis = np.asarray(self.unit_vector(rot_axis))
            a = math.cos(theta / 2.0)
            b, c, d = -rot_axis * math.sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            rot_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                   [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                   [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
            return rot_matrix @ vector
        else:
            return vector

    @property
    def prediction_horizon(self):
        """

        Returns
        -------
            prediction_horizon: int
                length of calculate trajectory in number of timesteps
        """
        return self.__prediction_horizon
