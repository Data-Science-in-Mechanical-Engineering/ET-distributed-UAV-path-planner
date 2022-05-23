import numpy as np
import matplotlib.pyplot as plt
import diversipy.hycusampling as hycusampling
import random
import math
from scipy.optimize import linear_sum_assignment


class Initializer():
    def __init__(self, testbed, rng_seed):

        self.__testbed = testbed

        # calculate the cuboids edges along the x,y,z plane
        self.__testbed_edges = self.__testbed.edges()

        # calculate edges that define a cuboid along the x,y,z axis
        self.__x_dimension = [self.__testbed_edges[0][0], self.__testbed_edges[1][0]]
        self.__y_dimension = [self.__testbed_edges[0][1], self.__testbed_edges[1][1]]
        self.__z_dimension = [self.__testbed_edges[0][2], self.__testbed_edges[1][2]]

        self.__seed = rng_seed
        np.random.seed(self.__seed)

    def initialize(self, init_type, dist_to_wall, num_points, min_dist, scaling_factor=1):

        """ initializes drone and target positions randomly in the testbed

        Parameters:
            init_type: string
                type of initialization (circle, random, random_stratified)
            dist_to_wall: double
                distance that should be kept from the wall when generating the random positions
            num_points: int
                number of random points that should be generated for the random algorithm
        """

        INIT_XYZS = []
        INIT_TARGETS = []

        if init_type == 'random':

            for i in range(num_points):
                # calculate random point in the testbed with distance to the wall
                INIT_XYZS.append(
                    [random.uniform(self.__x_dimension[0] + dist_to_wall, self.__x_dimension[1] - dist_to_wall),
                     random.uniform(self.__y_dimension[0] + dist_to_wall, self.__y_dimension[1] - dist_to_wall),
                     random.uniform(self.__z_dimension[0] + dist_to_wall, self.__z_dimension[1] - dist_to_wall)])

                INIT_TARGETS.append(
                    [random.uniform(self.__x_dimension[0] + dist_to_wall, self.__x_dimension[1] - dist_to_wall),
                     random.uniform(self.__y_dimension[0] + dist_to_wall, self.__y_dimension[1] - dist_to_wall),
                     random.uniform(self.__z_dimension[0] + dist_to_wall, self.__z_dimension[1] - dist_to_wall)])

        elif init_type == 'circle':

            # radius of circle to keep distance from the wall
            R = min(abs(self.__x_dimension[0] - self.__x_dimension[1]) - 2 * dist_to_wall,
                    abs(self.__y_dimension[0] - self.__y_dimension[1]) - 2 * dist_to_wall)
            R = R / 2

            testbed_center = self.__testbed.center()

            angles = [2 * math.pi * i / num_points for i in range(num_points)]
            for angle in angles:
                INIT_XYZS.append([R * math.cos(angle) + testbed_center[0],
                                  R * math.sin(angle) + testbed_center[1],
                                  testbed_center[2]])

                # initialize targets on other side of the circle (angle shifted by pi)
                INIT_TARGETS.append([R * math.cos(angle + math.pi) + testbed_center[0],
                                     R * math.sin(angle + math.pi) + testbed_center[1],
                                     testbed_center[2]])


        elif init_type == 'random_stratify':
            spawn_cube = ([edge + dist_to_wall for edge in self.__testbed_edges[0]],
                          [edge - dist_to_wall for edge in self.__testbed_edges[1]])

            random_points = hycusampling.maximin_reconstruction(num_points, 3)

            random_points = self._scale_points_to_testbed(random_points, spawn_cube)

            # randomly select subcubes (each can only be selected once)
            point_candidates_for_pos_initialization = [i for i in range(num_points)]
            point_candidates_for_target_initialization = [i for i in range(num_points)]
            points_for_pos_initialization = []
            points_for_target_initialization = []
            for i in range(num_points):
                point_chosen_for_pos_init = random.choice(point_candidates_for_pos_initialization)
                point_candidates_for_pos_initialization.remove(point_chosen_for_pos_init)
                points_for_pos_initialization.append(point_chosen_for_pos_init)

                point_chosen_for_target_init = random.choice(point_candidates_for_target_initialization)
                point_candidates_for_target_initialization.remove(point_chosen_for_target_init)
                points_for_target_initialization.append(point_chosen_for_target_init)

            for i in points_for_pos_initialization:
                INIT_XYZS.append(random_points[i])

            for i in points_for_target_initialization:
                INIT_TARGETS.append(random_points[i])


        elif init_type == 'random_stratify_max_dist':
            # maximises the total distance between spawn and target locations

            spawn_cube = ([edge + dist_to_wall for edge in self.__testbed_edges[0]],
                          [edge - dist_to_wall for edge in self.__testbed_edges[1]])

            try:
                drone_locations = self.generate_points__min_distance(num_points, spawn_cube, min_dist=min_dist,
                                                                     max_iter=20000, scaling_factor=scaling_factor)
            except ValueError:
                drone_locations = hycusampling.maximin_reconstruction(num_points, 3)
                drone_locations = self._scale_points_to_testbed(drone_locations, spawn_cube)

            try:
                target_locations = self.generate_points__min_distance(num_points, spawn_cube, min_dist=min_dist,
                                                                      max_iter=200000, scaling_factor=scaling_factor)

            except ValueError:
                target_locations = hycusampling.maximin_reconstruction(num_points, 3)
                target_locations = self._scale_points_to_testbed(target_locations, spawn_cube)


            """
            # create random starting positions and scale them
            drone_locations = hycusampling.maximin_reconstruction(self.__num_drones, 3)
            drone_locations = self._scale_points_to_testbed(drone_locations, spawn_cube)

            # create random target positions and scale them

            """

            # points_in_subcubes = self._sort_points_by_subcube(random_points, subcubes)

            # create weight matrix
            weights = []
            for i in range(len(drone_locations)):
                weights.append([])
                for j in range(len(target_locations)):
                    dist = np.linalg.norm(drone_locations[i] - target_locations[j])
                    weights[i].append(1 / dist)

            # find maximum total distance
            row_ind, col_ind = linear_sum_assignment(weights)

            for i in range(len(row_ind)):
                INIT_XYZS.append(drone_locations[row_ind[i]])
                INIT_TARGETS.append(target_locations[col_ind[i]])

        # convert to numpy arrays
        INIT_XYZS = np.array(INIT_XYZS)
        INIT_TARGETS = np.array(INIT_TARGETS)

        return INIT_XYZS, INIT_TARGETS

    def _scale_points_to_testbed(self, points, cube_for_scaling):
        """ scale random points in unit cube to fit inside cube_for_scaling

        Parameters:
            points: list or 3D np.array
                List of points that should be scaled to dimensions of cube_for_scaling
            cube_for_scaling: tuple
                cube with dimensions to scale position of points"""

        origin = cube_for_scaling[0]
        dx = cube_for_scaling[1][0] - cube_for_scaling[0][0]
        dy = cube_for_scaling[1][1] - cube_for_scaling[0][1]
        dz = cube_for_scaling[1][2] - cube_for_scaling[0][2]
        scaling_matrix = np.diag([dx, dy, dz])

        # scale each point
        for i in range(len(points)):
            points[i] = np.dot(points[i], scaling_matrix)
            points[i] = points[i] + origin

        return points

    def _divide_cube_in_subcubes(self, cube):
        """ divide a given cube into 8 equally sized smaller cubes

        Parameters:
            cube: tuple
                cube to divide into smaller cubes
            """
        origin = cube[0]
        dx = cube[1][0] - cube[0][0]
        dy = cube[1][1] - cube[0][1]
        dz = cube[1][2] - cube[0][2]

        subcubes = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    new_cube_origin = [sum(x) for x in zip(origin, [dx / 2 * x, dy / 2 * y, dz / 2 * z])]
                    subcubes.append((new_cube_origin, [sum(x) for x in zip(new_cube_origin, [dx / 2, dy / 2, dz / 2])]))

        return subcubes

    def _sort_points_by_subcube(self, points, subcubes):
        """ sort the points into the cube they are in

        Parameters:
            points: List of 3D np.array
                points to sort in correct cube
            subcubes: List of tubples
                cubes to sort points into """

        # create empty list
        point_locations = [None] * len(subcubes)

        # check for every subcube what point is inside
        for i in range(len(subcubes)):
            point_locations[i] = []
            for j in range(len(points)):
                if self._check_point_in_cube(points[j], subcubes[i]):
                    point_locations[i].append(points[j])

        return point_locations

    def _check_point_in_cube(self, point, cube):
        """ check if a given point is inside a cube
        Parameters:
            point: np.array
                point to check
            cube: tuple """

        point_in_lower_bounds = all([x1 - x2 >= 0 for (x1, x2) in zip(point, cube[0])])
        point_in_upper_bounds = all([x1 - x2 < 0 for (x1, x2) in zip(point, cube[1])])

        return point_in_lower_bounds and point_in_upper_bounds

    def generate_points__min_distance(self, num_points, shape, min_dist, max_iter, scaling_factor):
        """ generates random points with a minimum distance from each other """

        scaling_matrix = np.diag([1, 1, 1.0 / float(scaling_factor)])

        min_pos = [shape[0][0], shape[0][1], shape[0][2]]
        max_pos = [shape[1][0], shape[1][1], shape[1][2]]

        for n in range(max_iter):
            points = [np.random.uniform(min_pos, max_pos) for i in range(num_points)]
            valid_configuration = True
            for j in range(num_points):
                if not valid_configuration:
                    break

                for i in range(num_points):
                    if i <= j:
                        continue
                    dist = np.linalg.norm((points[j] - points[i]) @ scaling_matrix)
                    if dist < min_dist:
                        valid_configuration = False
                        break

            if valid_configuration:
                return points

        raise ValueError(
            'No configuration of random samples in the specified cube found. Try reducing the minimum distance')

