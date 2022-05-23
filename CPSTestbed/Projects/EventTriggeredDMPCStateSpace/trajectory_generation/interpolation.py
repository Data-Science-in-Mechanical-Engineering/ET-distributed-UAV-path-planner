"""
Contains code for trajectory interpolation
"""

import scipy.interpolate as interpolate
from scipy.special import binom
import numpy as np
from dataclasses import dataclass
from abc import ABC


@dataclass
class TrajectoryCoefficients:
    """trajectory coeffiients

    Parameters
    ----------
        coefficients: np.array, shape(prediction_horizon, order, 3)
        valid: bool
            true if coefficients represent a valid trajectory or not (because for example trajectory is not calculated
            yet or optimization problem is not feasible)
        alternative_trajectory: np.array, shape(prediction_horizon, 3)
            if valid is false, then in this fiela an alternative that can be used should be stored
    """
    coefficients: any
    valid: bool
    alternative_trajectory: any


class Interpolation(ABC):
    """this abstract class represents the Interpolation for the trajectory generation"""
    def __init__(self, breakpoints,
                 shape_optimization_variables, dimension_trajectory):
        """

        Parameters
        ----------
            breakpoints: np.array, shape(k_start+1,)
                breakpoints of the interpolation. The trajectory has k_start optimization variables
            shape_optimization_variables: tuple
                shape of optimization variables at one breakpoint, there exist k_start variables in total.
            dimension_trajectory: int
                dimension of trajectory (typically 3 or 2)
        """
        self.__num_optimization_variables = len(breakpoints) - 1
        for i in list(shape_optimization_variables):
            self.__num_optimization_variables = self.__num_optimization_variables * i

        self.__num_intervals = len(breakpoints) - 1
        self.__breakpoints = breakpoints

        self.__shape_optimization_variables = shape_optimization_variables
        self.__dimension_trajectory = dimension_trajectory

        self.__dim = dimension_trajectory

    @property
    def num_intervals(self):
        return self.__num_intervals

    @property
    def dimension(self):
        """
        Returns
        -------
            dimension_trajectory: int
                dimension of trajectory (typically 3 or 2)
        """
        return self.__dimension_trajectory

    @property
    def shape_optimization_variables(self):
        """
        Returns
        -------
            shape_optimization_variables: tuple
                shape of optimization variables at one breakpoint
        """
        return self.__shape_optimization_variables

    @property
    def breakpoints(self):
        """
        Returns
        -------
            breakpoints
        """
        return self.__breakpoints

    @property
    def num_optimization_variables(self):
        """
        Returns
        -------
            num_optimization_variables: int
                number of optimization variables
        """
        return self.__num_optimization_variables

    @property
    def num_equality_constraints(self):
        """
        Returns
        -------
           num_equality_constraints: int
                number of equality constraints that are needed internally in the interpolation. (e.g. that are needed
                such that the interpolation is continuous/differentiable etc.)
        """
        return 0

    @property
    def equality_constraint(self):
        """
        generates equality constraint A*x = b
        Returns
        -------
            A: np.array, shape(num_equality_constraints, num_optimization_variables)
            b: np.array, shape(num_equality_constraints,)
        """
        return None, None

    def interpolate_vector(self, x_vector, polynomial_coefficients, derivative_order=0):
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

        Returns
        -------
            y_vector: np.array, shape(l, dim)
                interpolation values at x_vector. If the coefficients are not valid, it simply clips the return
                valid point to the lowest nearest.
        """

        y_vector = np.zeros((len(x_vector), self.dimension))
        for i in range(0, len(x_vector)):
            y_vector[i] = self.interpolate(x_vector[i], polynomial_coefficients, derivative_order)
        return y_vector

    def interpolate(self, x, polynomial_coefficients, derivative_order=0):
        """
        get value of interpolation at ont point.

        Parameters
        ----------
            x: float
                value to evaluate at
            derivative_order: int
                order of derivative
            polynomial_coefficients: TrajectoryCoefficients
                trajectory coefficients

        Returns
        -------
            y: np.array, shape(dim,)
                interpolation value at x. If the coefficients are not valid, it simply clips the return
                valid point to the lowest nearest, if derivative is zero, otherwise o.
        """
        return np.zeros((self.dimension_trajectory,))

    def get_trajectory_vector_matrix(self, x_vector, derivative_order=0):
        """calculates the matrix that allows to calculate the trajectory by multiplying it with the optimization variables

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
        return np.zeros((len(x_vector), ) + self.shape_optimization_variables)

    def find_interval(self, x):
        """find corresponding interval in breakpoints

        Parameters
        ----------
            x: float
                timepoint

        Returns
        -------
            num_interval: int
                number of interval x lies into.
        """
        # find interval
        index_interval = 0
        while x > self.breakpoints[index_interval + 1]:
            index_interval = index_interval + 1
            # last interval reached?
            if index_interval == self.num_intervals - 1:
                break
        return index_interval


class PiecewiseBernsteinPolynomial(Interpolation):
    """
    represents a piecewise bernstein polynomial and its derivatives.
    """

    def __init__(self, breakpoints, dimension_trajectory, order):
        """
        constructor

        Parameters
        ----------
        breakpoints: np.array, shape(m+1, 1)
            breakpoints of the polynomial, m is the number of intervals
        dimension_trajectory: int
            dimension of trajectory
        order: int
            order of bernstein polynom (maximum polynomial order is order-1)
        """
        super().__init__(breakpoints, (order, dimension_trajectory), dimension_trajectory)
        self.__breakpoints = breakpoints

        self.__order = order  # maximum polynomial order is k_start-1

    @property
    def order(self):
        return self.__order

    def interpolate(self, x, polynomial_coefficients, derivative_order=0):
        """
        get value of interpolation at ont point.

        Parameters
        ----------
            x: float
                value to evaluate at
            derivative_order: int
                order of derivative
            polynomial_coefficients: TrajectoryCoefficients
                trajectory coefficients

        Returns
        -------
            y: np.array, shape(dim,)
                interpolation value at x. If the coefficients are not valid, it simply clips the return
                valid point to the lowest nearest, if derivative is zero, otherwise o.
        """

        basis_functions, index_interval = self.get_basis_functions(x, derivative_order)
        if polynomial_coefficients.valid:
            return basis_functions @ polynomial_coefficients.coefficients[index_interval, :, :]
        return polynomial_coefficients.alternative_trajectory[index_interval, :] if derivative_order == 0 else np.zeros((3,))

    def get_trajectory_vector_matrix(self, x_vector, derivative_order=0):
        """calculates the matrix that allows to calculate the trajectory by multiplying it with polynomial_coefficients

        when putting all polynomial coefficients into one big vector, one can get a big vector of the points of the
        trajectory at timepoints stored in x_vector by multipling the vecotr of polynomial coefficients with the matrix
        this method returns

        Parameters
        ----------
            x_vector: np.array, shape(l,)
                time points the trajectory should be calculated at
            derivative_order: int
                order of derivative d/dx
        Returns
        -------
            matrix: np.array, shape(dim*l, k_start*m*dim)
        """
        length_trajectory = len(x_vector)

        matrix = np.zeros((self.dimension*length_trajectory, self.num_optimization_variables))

        for i in range(0, length_trajectory):
            basis_functions, index = self.get_basis_functions(x_vector[i], derivative_order)
            # generate matrizes
            for j in range(0, self.dimension):
                for k in range(0, self.order):
                    x = i*self.dimension + j
                    y = index*self.dimension*self.order + k*self.dimension + j
                    matrix[i*self.dimension + j, index*self.dimension*self.order + k*self.dimension + j] = \
                        basis_functions[k]

        return matrix

    def get_basis_functions(self, x, derivative_order=0):
        """returns basis function at time x in an array
        Parameters
        ----------
        x: float
            value the piecewise polynomial should be evaluated at
        derivative_order: int
            order of derivative d/dx

        Returns
        -------
        basis_functions: np.array shape(k_start, )
            array contains the basis functions b(.,k_start,t(x)), where k_start is the order, t(x) in [0, 1] and
            t(x) = (x-breakpoints(index_interval)/(breakpoints(index_interval+1)-breakpoints(index_interval))).
        index_interval: int
            index of interval x belongs to.
        """
        index_interval = self.find_interval(x)

        # calculate basis functions
        t = (x-self.breakpoints[index_interval]) / (self.breakpoints[index_interval+1]-self.breakpoints[index_interval])
        if t > 1:
            t = 1
        basis_functions = [self.basis_function(a, t, self.order-1, derivative_order)/((self.breakpoints[1]-self.breakpoints[0])**derivative_order) for a in range(0, self.order)]
        return basis_functions, index_interval

    def get_basis_functions_raw(self, t, derivative_order=0):
        """
        calculates all basis functions at one time t in [0;1]

        Parameters
        ----------
        t: float
             between 0 and 1, point to evaluate at
        derivative_order: int
            order of derivative to calculate at (d/dt attention not d/dx)

        Returns
        -------
        basis_functions: np.array, shape(k_start+1,)
            all basis functions in a vector
        """
        return [self.basis_function(i, t, self.order-1, derivative_order) for i in range(0, self.order)]

    def basis_function(self, degree, t, order, derivative_order=0):
        """
        calculates basis function
        Parameters
        ----------
            degree: int
                degree of polynom to evaluate
            t: float
                between 0 and 1, point to evaluate at
            order: int
                order of bernstein polynom
            derivative_order: int
                order of derivative to calculate (d/dt attention not d/dx)
        Returns
        -------
            value: float
                derivative_order'th derivative of basis function at time t
        """
        if degree == -1 or order == degree - 1:
            return 0
        if derivative_order == 0:
            return binom(order, degree) * (t**degree) * ((1-t)**(order-degree))
        else:
            return order*(self.basis_function(degree-1, t, order-1, derivative_order-1) - self.basis_function(degree, t, order-1, derivative_order-1))

    @property
    def num_equality_constraints(self):
        """
        Returns
        -------
           num_equality_constraints: int
                number of equality constraints that are needed internally in the interpolation. (e.g. that are needed
                such that the interpolation is continuous/differentiable etc.)
        """
        return 3*self.dimension * (self.num_intervals - 1)

    @property
    def equality_constraint(self):
        """
        generates equality constraint A*x = b, x are the flattened optimization variables. This constraint is important
        such that the resulting trajectory is continuous.
        Returns
        -------
            A: np.array, shape(num_equality_constraints, num_optimization_variables)
            b: np.array, shape(num_equality_constraints,)
        """
        left_side_matrix = np.zeros((self.num_equality_constraints, self.num_optimization_variables))
        right_side_matrix = np.zeros((len(left_side_matrix, )))
        # build constraints for continuity in position, speed, acceleration
        for i in range(0, 3):
            left_side_matrix[i*self.dimension * (self.num_intervals - 1):
                             (i+1)*self.dimension * (self.num_intervals - 1), :] = \
                self.build_continuity_constraint_matrix(i)
        return left_side_matrix, right_side_matrix

    def build_continuity_constraint_matrix(self, derivative_order=0):
        """builds constraint matrix
        Builds constraint matrix in the form Aeq = ..., where Aeq is such that multiplied with the coefficients
        such that it forms the terms.  p(0, 1)-p(1, 0), ..., p(N-1, 1)-p(N, 0)

        This matrix can be used to build start and final equality constraints on the trajectory and
        enforce that it is continuous

        Parameters
        ----------
            derivative_order: int
                derivative order the constraint matrix should be calculated

        Returns
        -------
            matrix: np.array, shape(dim*(m+1), dim*k_start*m)
                constraint matrix
        """

        matrix = np.zeros((self.dimension * (self.num_intervals - 1), self.num_optimization_variables))

        # first and last entry are special
        basis_functions_0 = self.get_basis_functions_raw(0, derivative_order)/((self.breakpoints[1]-self.breakpoints[0])**derivative_order)
        basis_functions_1 = self.get_basis_functions_raw(1, derivative_order)/((self.breakpoints[1]-self.breakpoints[0])**derivative_order)
        for i in range(1, self.num_intervals):
            for j in range(0, self.dimension):
                for k in range(0, self.order):
                    matrix[self.dimension*(i-1) + j, self.order*self.dimension*i + k*self.dimension + j] = \
                        basis_functions_0[k]
                    matrix[self.dimension*(i-1) + j, self.order * self.dimension * (i-1) + k * self.dimension + j] = \
                        -basis_functions_1[k]
        return matrix
