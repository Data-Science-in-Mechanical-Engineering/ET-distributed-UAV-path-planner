import copy
import sys
import os

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from CPSTestbed.network import network as net

import CPSTestbed.useful_scripts.cuboid as cuboid
import CPSTestbed.useful_scripts.initializer as initializer
import CPSTestbed.useful_scripts.custom_logger as custom_logger

import CPSTestbed.Projects.EventTriggeredDMPCStateSpace.drone_agent as da
import CPSTestbed.Projects.EventTriggeredDMPCStateSpace.trajectory_generation.trajectory_generation as tg

import time
import argparse
import math
import random
import numpy as np
import pybullet as p
import pickle
import gc


class Simulation:
    """
    defines a simulation instance

    Methods:
        run: runs the simulation
    """
    def __init__(self, ARGS):
        """constructor

        Parameters:
            ARGS:
                options for the simulation

        """
        self.__id = ARGS.sim_id
        # copy ARGS in case it gets modified later
        self.__ARGS = copy.deepcopy(ARGS)

        # initialize the simulation
        self.__testbed = cuboid.Cuboid(np.array([0, 0, 1]), np.array([ARGS.testbed_size[0], 0, 0]),
                                       np.array([0, ARGS.testbed_size[1], 0]),
                                       np.array([0, 0, ARGS.testbed_size[2]]))

        self.__INIT_XYZS = ARGS.INIT_XYZS
        self.__INIT_TARGETS = ARGS.INIT_TARGETS

        # self.__INIT_XYZS = np.array([[0.5, 1, 2], [4, 1.4, 2]])
        # self.__INIT_TARGETS = np.array([[4, 1, 2], [0.5, 1.4, 2]])
        self.__INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])  # initial rotation
        self.__INIT_VELOCITY = np.array([[0, 0, 0]])  # initialize velocities
        self.__AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz / ARGS.control_freq_hz) if ARGS.aggregate else 1

        # create environment without video capture
        self.__env = CtrlAviary(drone_model=self.__ARGS.drone,
                                num_drones=self.__ARGS.num_drones,
                                initial_xyzs=self.__INIT_XYZS,
                                initial_rpys=self.__INIT_RPYS,
                                physics=self.__ARGS.physics,
                                neighbourhood_radius=10,
                                freq=self.__ARGS.simulation_freq_hz,
                                aggregate_phy_steps=self.__AGGR_PHY_STEPS,
                                gui=self.__ARGS.gui,
                                record=self.__ARGS.record_video,
                                obstacles=self.__ARGS.obstacles,
                                user_debug_gui=self.__ARGS.user_debug_gui
                                )

        # Obtain the PyBullet Client ID from the environment
        self.__PYB_CLIENT = self.__env.getPyBulletClient()

        # Remove empty menus
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Initialize the logger
        self.__desample = 10
        self.__logger = custom_logger.CustomLogger(ARGS=self.__ARGS,
                                                   logging_freq_hz=int(
                                                       self.__ARGS.simulation_freq_hz / self.__AGGR_PHY_STEPS / self.__desample),
                                                   num_drones=self.__ARGS.num_drones,
                                                   duration_sec=self.__ARGS.duration_sec)


        # Initialize the controllers for each drone
        if self.__ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            self.__ctrl = [DSLPIDControl(drone_model=self.__ARGS.drone) for i in range(self.__ARGS.num_drones)]
        elif ARGS.drone in [DroneModel.HB]:
            self.__ctrl = [SimplePIDControl(drone_model=self.__ARGS.drone) for i in range(self.__ARGS.num_drones)]

        # initialize the network
        self.__agents = []
        self.__computing_agents = []
        delta_t = 1.0 / ARGS.communication_freq_hz
        collision_constraint_sample_points = np.linspace(0, ARGS.prediction_horizon * delta_t,
                    int(((ARGS.prediction_horizon) * 2+1)))
        collision_constraint_sample_points = collision_constraint_sample_points[1:]
        trajectory_generator_options = tg.TrajectoryGeneratorOptions(
            # different amount of sample point to increase resolution
            objective_function_sample_points=np.linspace(delta_t, ARGS.prediction_horizon * delta_t,
                                                         int((ARGS.prediction_horizon * 1))),
            state_constraint_sample_points=np.linspace(delta_t, ARGS.prediction_horizon * delta_t,
                                                       int((ARGS.prediction_horizon * 1))),
            collision_constraint_sample_points=collision_constraint_sample_points,
            weight_state_difference=np.eye(3) * 1,
            weight_state_derivative_1_difference=np.eye(3) * 0.01,
            weight_state_derivative_2_difference=np.eye(3) * 5,
            weight_state_derivative_3_difference=np.eye(3) * 0.01,
            max_speed=np.array([1.0, 1.0, 1.0]),
            max_position=np.array(self.__testbed.edges()[1]),  # np.array([10.0, 10.0, 100.0]),
            max_acceleration=np.array([2.0, 2.0, 2.0]),
            max_jerk=np.array([7.0, 7.0, 7.0]),
            min_position=np.array(self.__testbed.edges()[0]),  # np.array([-10.0, -10.0, 0.5]),
            r_min=self.__ARGS.r_min,
            optimization_variable_sample_time=delta_t / 2.0,  # can be tuned
            num_drones=ARGS.num_drones,
            skewed_plane_BVC=ARGS.skewed_plane_BVC,
            use_qpsolvers=ARGS.use_qpsolvers,
            downwash_scaling_factor=ARGS.downwash_scaling_factor,
            use_soft_constraints=ARGS.use_soft_constraints,
            guarantee_anti_collision=ARGS.guarantee_anti_collision,
            soft_constraint_max=ARGS.soft_constraint_max,
            weight_soft_constraint=ARGS.weight_soft_constraint
        )

        slot_group_trajectory = net.SlotGroup(0, False, self.__ARGS.num_computing_agents)

        agent_ids = {}
        for i in range(0, ARGS.num_drones):
            agent = da.RemoteDroneAgent(ID=i,
                                        slot_group_planned_trajectory_id=slot_group_trajectory.id,
                                        init_position=self.__INIT_XYZS[i],
                                        target_position=self.__INIT_TARGETS[i],
                                        communication_delta_t=delta_t,
                                        trajectory_generator_options=trajectory_generator_options,
                                        prediction_horizon=ARGS.prediction_horizon,
                                        order_interpolation=ARGS.interpolation_order)

            agent_ids[i] = i
            self.__agents.append(agent)

        for i in range(ARGS.num_drones, ARGS.num_drones + min(ARGS.num_computing_agents, ARGS.num_drones)):
            computing_agent = da.ComputationAgent(ID=i, slot_group_planned_trajectory_id=slot_group_trajectory.id,
                                                  init_positions=self.__INIT_XYZS,
                                                  target_positions=self.__INIT_TARGETS,
                                                  agents_ids=agent_ids,
                                                  communication_delta_t=delta_t,
                                                  trajectory_generator_options=trajectory_generator_options,
                                                  prediction_horizon=ARGS.prediction_horizon,
                                                  num_computing_agents=ARGS.num_computing_agents,
                                                  offset=(i - ARGS.num_drones) * int(
                                                      ARGS.num_drones / max((ARGS.num_computing_agents), 1)),
                                                  use_event_trigger=ARGS.event_trigger,
                                                  alpha_1=ARGS.alpha_1,
                                                  alpha_2 = ARGS.alpha_2,
                                                  alpha_3=ARGS.alpha_3
                                                  )
            self.__agents.append(computing_agent)
            self.__computing_agents.append(computing_agent)

        self.__network = net.Network(self.__agents)
        self.__network.add_slot_group(slot_group_trajectory)

    def run(self):
        """
        runs the simulation

        Return:
            simulation_result: logger
                result of simulation including:
                    - success: bool
                    - state of drones: np.array
                    - INIT_XYZS: np.array
                    - INIT_TARGETS: np.array
        """
        COM_EVERY_N_STEPS = int(np.floor(self.__env.SIM_FREQ / self.__ARGS.communication_freq_hz))
        CTRL_EVERY_N_STEPS = int(np.floor(self.__env.SIM_FREQ / self.__ARGS.control_freq_hz))
        action = {str(i): np.array([0, 0, 0, 0]) for i in range(self.__ARGS.num_drones)}
        START = time.time()
        critical_dist_to_target = 0.05

        num_image = 0
        print('Starting Simulation No. ' + str(self.__id))
        for ind in range(0, 1):
            pos_last = -1
            ball_position_estimator = None
            desample_time = 0
            for i in range(0, int(self.__ARGS.duration_sec * self.__env.SIM_FREQ), self.__AGGR_PHY_STEPS):
                #### Step the simulation ###################################
                obs, reward, done, info = self.__env.step(action)

                if i % COM_EVERY_N_STEPS == 0:
                    # the current position should be set before the network transmits its data (in reality the measurments
                    # are taken after the end of communication round)

                    all_targets_reached = True
                    for j in range(0, self.__ARGS.num_drones):
                        self.__agents[j].position = obs[str(j)]["state"][0:3]

                        dist_to_target = np.linalg.norm(self.__agents[j].position - self.__agents[j].target_position)
                        # set transition time only, if it's not already set to not override it
                        reached_target = self.__agents[j].target_reached
                        if dist_to_target < critical_dist_to_target and not reached_target:
                            self.__agents[j].transition_time = i / self.__env.SIM_FREQ
                            self.__agents[j].target_reached = True
                        elif dist_to_target >= critical_dist_to_target:
                            self.__agents[j].transition_time = None
                            self.__agents[j].target_reached = False

                        all_targets_reached = all_targets_reached and reached_target

                    # check if drones have crashed
                    for j in range(0, self.__ARGS.num_drones):
                        # for n in range(0, self.__ARGS.num_drones):
                        scaling_matrix = np.diag([1, 1, 1.0 / float(self.__ARGS.downwash_scaling_factor_crit)])
                        if any([np.linalg.norm(
                                (self.__agents[j].position - self.__agents[n].position)@scaling_matrix) < self.__ARGS.r_min_crit for n in
                                range(0, self.__ARGS.num_drones) if n != j]):
                            self.__agents[j].crashed = True

                    # step network
                    self.__network.step()

                # do low level control
                if i % CTRL_EVERY_N_STEPS == 0:

                    next_state = np.zeros((self.__ARGS.num_drones, 9))
                    #### Compute control for the current way point #############
                    for j in range(self.__ARGS.num_drones):
                        control_interval = 1.0 / self.__ARGS.control_freq_hz
                        next_state[j, :] = np.copy(self.__agents[j].next_planned_state(control_interval))
                        action[str(j)], _, _ = self.__ctrl[j].computeControlFromState(
                            control_timestep=CTRL_EVERY_N_STEPS * self.__env.TIMESTEP,
                            state=obs[str(j)]["state"],
                            target_pos=next_state[j, :3],
                            target_vel=next_state[j, 3:6],
                            target_rpy=self.__INIT_RPYS[j, :]
                        )

                #### Log the simulation ####################################
                if desample_time % self.__desample == 0:
                    for j in range(self.__ARGS.num_drones):
                        self.__logger.log(drone=j, timestamp=i / self.__env.SIM_FREQ, state=obs[str(j)]["state"],
                                          control=np.hstack((next_state[j, :3], next_state[j, 3:6], np.zeros(6))))

                desample_time += 1
                #### Sync the simulation ###################################
                if self.__ARGS.gui:
                    sync(i, START, self.__env.TIMESTEP)

                #### Stop the simulation, if all drones reached their targets or have crashed
                stop = True
                stop2 = False
                for j in range(self.__ARGS.num_drones):
                    stop = stop and (self.__agents[j].crashed or self.__agents[j].target_reached)
                    if self.__agents[j].crashed:
                        stop2 = True
                if (stop or stop2) and self.__ARGS.abort_simulation:
                    break

        # log results of simulation
        for j in range(self.__ARGS.num_drones):
            self.__logger.log_static(drone=j, successful=(not self.__agents[j].crashed)
                                                         and not (self.__agents[j].transition_time is None),
                                     transition_time=self.__agents[j].transition_time,
                                     distance_travelled=0,
                                     target=self.__agents[j].target_position, crashed=self.__agents[j].crashed)

        av_transition_time = np.mean([self.__agents[j].transition_time for j in range(self.__ARGS.num_drones) if self.__agents[j].transition_time is not None])
        crashed = sum([self.__agents[j].crashed for j in range(self.__ARGS.num_drones)])
        print('Simulation ' + str(self.__id) + ' done. Average Transition Time: ' + str(np.round(av_transition_time,2)) + 's. ' + str(crashed) + ' Drones crashed.')

        # save the logged data
        self.save(self.__logger)

        # delete the logger
        self.__logger = None
        del self.__logger
        gc.collect()

        return True

    def save(self, simulation_logger):
        with open(
                os.path.join(self.__ARGS.path, "simulation_result-" + str(self.__ARGS.num_drones) + "_drones_simnr_" + str(self.__id) + ".pkl"), 'wb') \
                as out_file:
            pickle.dump(simulation_logger, out_file)
