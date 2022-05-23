from CPSTestbed.Projects.EventTriggeredAntiCollision.trajectory_generation.interpolation import Interpolation, \
    TrajectoryCoefficients

import numpy as np


class StateSpace(Interpolation):
    """ This class represents a statespace model """

    def __init__(self, breakpoints, dimension_trajectory, num_states, sampling_time, model_order):
        """

        Parameters
        ----------
            breakpoints: np.array, shape(k_start+1,)
                breakpoints of the interpolation. The trajectory has k_start optimization variables
            dimension_trajectory: int
                dimension of trajectory (typically 3 or 2)
            num_states: int
                number of states
            num_inputs: int
                number of inputs

        """
        super().__init__(breakpoints, (len(breakpoints), dimension_trajectory), dimension_trajectory)
        self.__num_states = num_states
        self.__num_inputs = dimension_trajectory
        self.__sampling_time = sampling_time
        self.__num_optimization_variables = self.__num_inputs * (len(breakpoints)-1)
        self.__breakpoints = breakpoints
        self.__model_order = model_order

    def discrete_state_matrix(self, dT):
        """ calculates the discrete A matrix of the statespace model """
        pass

    def discrete_input_matrix(self, dT):
        """ calculates the discrete B matrix of the statespace model """
        pass


    def get_input_trajectory_vector_matrix(self, x_vector, derivative_order=0):
        """calculates the matrix that allows to calculate the trajectory from the input by multiplying it with the
        optimization variables

        when putting all polynomial coefficients into one big vector, one can get a big vector of the points of the
        trajectory at timepoints stored in x_vector by multiplying the vecotr of polynomial coefficients with the matrix
        this method returns

        Parameters
        ----------
            x_vector: np.array, shape(l,)
                time points the trajectory should be calculated at
            derivative_order: int
                order of derivative d/dx
        Returns
        -------
            matrix: np.array, shape(dimension_trajectory*len(x_vector), num_optimization_variables)
                when the optimization variables are put into a matrix with (breakpoint, shape_optimization_variables) and then
                flattened with C-style, we can get the trajectory by [y(x_vector[0], ...)] =
                matrix@flattened_optimization_variables.
        """
        if derivative_order > self.__model_order:
            return None
        elif derivative_order == self.__model_order:
            return np.eye(len(x_vector))

        A_0 = self.discrete_state_matrix(self.__sampling_time)
        B_0 = self.discrete_input_matrix(self.__sampling_time)
        psi = np.zeros((self.dimension, self.__num_states))
        psi[:, derivative_order * self.dimension:(derivative_order + 1) * self.dimension] = np.eye(self.dimension)

        matrix = np.zeros((self.dimension * len(x_vector), self.dimension * (len(self.__breakpoints)-1)))
        for i in range(len(x_vector)):
            full_segments = int(x_vector[i] / self.__sampling_time)  # TODO: add prev integration timestep
            dT = x_vector[i] - full_segments * self.__sampling_time

            for j in range(full_segments):
                matrix[i * self.dimension:(i + 1) * self.dimension,
                j * self.dimension:(j + 1) * self.dimension] = psi @ self.discrete_state_matrix(
                    dT) @ np.linalg.matrix_power(
                    A_0, full_segments - j - 1) @ B_0

            if dT > 0:
                matrix[i * self.dimension:(i + 1) * self.dimension,
                full_segments * self.dimension:(full_segments + 1) * self.dimension] = psi @ self.discrete_input_matrix(dT)

        return matrix

    def get_state_trajectory_vector_matrix(self, x_vector, derivative_order=0):

        A = self.discrete_state_matrix(0)
        psi = np.zeros((self.dimension, self.__num_states))
        psi[:, derivative_order * self.dimension:(derivative_order + 1) * self.dimension] = np.eye(self.dimension)

        state_matrix = np.tile(np.zeros((self.dimension, self.__num_states)), (len(x_vector), 1))

        lastTime = 0
        for i in range(len(x_vector)):
            curr_time = x_vector[i]
            dT = curr_time - lastTime
            k0 = int(lastTime / self.__sampling_time)
            for j in range(int(dT / self.__sampling_time)):
                A = self.discrete_state_matrix(self.__sampling_time) @ A

            k_curr = k0 + int(dT / self.__sampling_time)
            dT = dT % self.__sampling_time
            A = self.discrete_state_matrix(dT) @ A
            state_matrix[i * self.dimension:(i + 1) * self.dimension, :] = psi @ A
            lastTime = curr_time

        return state_matrix

    def interpolate(self, x, polynomial_coefficients, derivative_order=None, integration_start=0, x0=None):
        """
                get value of interpolation at ont point. if derivative_order < model_order, the state will be integrated
                 at desired value. Assumption: u starts at t=0 and is evenly spaced with self.__sampling_time

                Parameters
                ----------
                    x: float
                        value to evaluate at
                    derivative_order: int
                        order of derivative
                    polynomial_coefficients: TrajectoryCoefficients
                        trajectory coefficients
                    integration_start: float
                        timestamp, at which the interpolation should start
                    x0: np.array, shape(num_states,)
                        state at start of integration

                Returns
                -------
                    y: np.array, shape(dim,)
                        interpolation value at x. If the coefficients are not valid, it simply clips the return
                        valid point to the lowest nearest, if derivative is zero, otherwise o.
                """
        t = np.clip(x, 0, self.__breakpoints[-1])
        integration_start = np.clip(integration_start, 0, self.__breakpoints[-1])

        psi = self.calc_psi(derivative_order)

        if x0 is None:
            x0 = np.zeros((self.__num_states,))
        x = x0
        if polynomial_coefficients.valid:
            u = polynomial_coefficients.coefficients
        else:
            u = polynomial_coefficients.alternative_trajectory

        k0 = int(integration_start / self.__sampling_time) # index at beginning of integration
        k_end = int(t / self.__sampling_time)  # index at desired time value
        if (integration_start % self.__sampling_time) >= 1e-15 and k_end > k0:
            dT_start = (k0 + 1) * self.__sampling_time - integration_start
            x = self.discrete_state_matrix(dT_start) @ x + self.discrete_input_matrix(dT_start) @ u[k0,:]
            k0 += 1

        full_segments = k_end - k0
        for j in range(full_segments):
            x = self.discrete_state_matrix(self.__sampling_time) @ x + self.discrete_input_matrix(self.__sampling_time) \
                @ u[k0 + j, :]

        if k_end == int(integration_start / self.__sampling_time):  # beginning and end of integration on same input sample
            dT_end = t - integration_start
        else:
            dT_end = t - (full_segments + k0) * self.__sampling_time  # length of integration interval
        if dT_end > 1e-15:
            k_curr = k0 + full_segments

            x = self.discrete_state_matrix(dT_end) @ x + self.discrete_input_matrix(dT_end) @ u[k_curr, :]

        return psi @ x

    def interpolate_vector(self, x_vector, polynomial_coefficients, derivative_order=None, x0=None, integration_start=0):
        """
                get value of interpolation at some points.

                Parameters
                ----------
                    x_vector: np.array, shape(l,)
                        time points the trajectory should be calculated at
                    derivative_order: int
                        order of derivative
                    polynomial_coefficients: TrajectoryCoefficients
                        trajectory coefficients
                    x0: np.array, shape(dim,)
                        initial states

                Returns
                -------
                    y_vector: np.array, shape(l, dim)
                        interpolation values at x_vector. If the coefficients are not valid, it simply clips the return
                        valid point to the lowest nearest.
                """
        psi = self.calc_psi(derivative_order)

        y_vector = np.zeros((len(x_vector), self.dimension))
        prev_timestep = integration_start
        for i in range(0, len(x_vector)):
            x0 = self.interpolate(x_vector[i], polynomial_coefficients, x0=x0, integration_start=prev_timestep)
            prev_timestep = x_vector[i]
            y_vector[i] = psi@x0
        return y_vector

    def calc_psi(self, derivative_order):
        """ calculates psi to select, which derivative of the state should be returned"""

        if derivative_order is None:
            for j in range(self.__model_order):
                psi = np.eye(self.__model_order * self.dimension)
        else:
            psi = np.zeros((self.dimension, self.__num_states))  # TODO: Check, if derivative_order < model order
            psi[:, derivative_order * self.dimension:(derivative_order + 1) * self.dimension] = np.eye(self.dimension)

        return psi

    @property
    def num_optimization_variables(self):
        """ returns the number of optimization variables """
        return self.__num_optimization_variables

    @property
    def breakpoints(self):
        """
        Returns
        -------
            breakpoints
        """
        return self.__breakpoints

class TripleIntegrator(StateSpace):
    """ This class represents a triple integrator statespace model """
    def __init__(self, breakpoints, dimension_trajectory, num_states, sampling_time):
        """

        Parameters
        ----------
            breakpoints: np.array, shape(k_start+1,)
                breakpoints of the integration. The trajectory has k_start optimization variables
            dimension_trajectory: int
                dimension of trajectory (typically 3 or 2)
            num_states: int
                number of states
            num_inputs: int
                number of inputs

        """
        super().__init__(breakpoints, dimension_trajectory, num_states, sampling_time, 3)


    def discrete_state_matrix(self, dT):
        """ calculates the discrete A matrix of the statespace model """
        A = np.array([[1, dT, 0.5 * dT ** 2], [0, 1, dT], [0, 0, 1]])
        return np.kron(A, np.eye(self.dimension))

    def discrete_input_matrix(self, dT):
        """ calculates the discrete B matrix of the statespace model """
        B = np.array([[1.0 / 6 * dT ** 3], [0.5 * dT ** 2], [dT]])
        return np.kron(B, np.eye(self.dimension))