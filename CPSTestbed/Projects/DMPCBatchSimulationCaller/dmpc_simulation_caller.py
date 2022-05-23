"""
Manages the calculation of multiple simulations. The parameters for the simulation are initialilzed. These parameters
are used to initialize the simulations by using the Simulation class.
"""
import copy
import multiprocessing
import sys
from joblib import Parallel, delayed

import time
import argparse
import math
import random
import numpy as np
import pybullet as p
# import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import multiprocessing as mp
import pickle as p
import gc

import shutil

# install gym-pybullet-drones
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool

import CPSTestbed.Projects.EventTriggeredDMPCStateSpace.dmpc_statespace_simulation as simulation
import CPSTestbed.useful_scripts.initializer as initializer
from CPSTestbed.useful_scripts.plotter import Plotter
import CPSTestbed.useful_scripts.cuboid as cuboid


def save(simulation_logger, ARGS, path, simulation_number):
    with open(
            path + "simulation_result-" + str(ARGS.num_drones) + "_drones" + str(simulation_number) + ".pkl", 'wb') \
            as out_file:
        pickle.dump([simulation_logger, ARGS], out_file)


def create_dir(path_to_logger):
    try:
        os.makedirs(path_to_logger)
    finally:
        return


def parallel_simulation_wrapper(ARGS_for_simulation):
    sim = simulation.Simulation(ARGS_for_simulation)
    sim.run()
    del sim
    gc.collect()
    return None


if __name__ == "__main__":
    #os.environ["OMP_NUM_THREADS"] = "1"
    #os.environ["MKL_NUM_THREADS"] = "1"
    #### Define and parse (optional) arguments for the script ##
    # !!!!!!!!!!!!!!!! Downwash simulation is unrealistic atm. I will contact the authors of the paper and discuss it
    parser = argparse.ArgumentParser(
        description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone', default="cf2x", type=DroneModel, help='Drone model (default: CF2X)', metavar='',
                        choices=DroneModel)

    parser.add_argument('--num_drones', default=[25], type=list,
                        help='List of number of drones to iterate over', metavar='')
    parser.add_argument('--physics', default="pyb_drag", type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--vision', default=False, type=str2bool, help='Whether to use VisionAviary (default: False)',
                        metavar='')
    parser.add_argument('--gui', default=False, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=False, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')

    parser.add_argument('--plot', default=False, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--plot_batch_results', default=True, type=str2bool,
                        help='Whether to plot the batch simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=False, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')

    parser.add_argument('--multiprocessing', default=True, type=str2bool,
                        help='Whether simulations run in parallel', metavar='')
    parser.add_argument('--log', default=True, type=str2bool,
                        help='Whether to log the simulations', metavar='')
    parser.add_argument('--log_path', default='', type=str,
                        help='Path for simulation logs', metavar='')
    parser.add_argument('--aggregate', default=True, type=str2bool,
                        help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles', default=False, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)',
                        metavar='')
    parser.add_argument('--control_freq_hz', default=60, type=int, help='Control frequency in Hz (default: 48)',
                        metavar='')
    parser.add_argument('--duration_sec', default=100, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--communication_freq_hz', default=3, type=int,
                        help='Communication frequency in Hz (default: 10)')
    parser.add_argument('--potential_function', default='chen', type=str,
                        help='Potential Field Function for Anti Collision')
    parser.add_argument('--drone_position_initialization_method', default='random_stratify_max_dist', type=str,
                        help='Method to initialize drone and target position')
    parser.add_argument('--testbed_size', default=[5, 5, 5], type=list,
                        help='Size of the area of movement for the drones')

    parser.add_argument('--abort_simulation', default=True, type=bool, help='Total number of simulations')

    parser.add_argument('--total_simulations', default=1000, type=int, help='Total number of simulations')
    parser.add_argument('--network_message_loss', default=[0], type=list,
                        help='List of message loss values of the communication network')
    parser.add_argument('--num_computing_agents', default=25, type=int, help='Number of computing agents for the '
                                                                            'DMPC Algorithm')
    parser.add_argument('--prediction_horizon', default=15, type=int, help='Prediction Horizon for DMPC')
    parser.add_argument('--interpolation_order', default=5, type=int, help='Order of the Bernstein Interpolation')
    parser.add_argument('--r_min', default=0.7, type=float, help='minimum distance to each Drone')
    parser.add_argument('--r_min_crit', default=0.2, type=float, help='minimum distance to each Drone')

    parser.add_argument('--use_soft_constraints', default=False, type=bool, help='')
    parser.add_argument('--guarantee_anti_collision', default=True, type=bool, help='')
    parser.add_argument('--soft_constraint_max', default=0.2, type=float, help='')
    parser.add_argument('--weight_soft_constraint', default=0.01, type=float, help='')

    parser.add_argument('--sim_id', default=0, type=int, help='ID of simulation, used for random generator seed')
    parser.add_argument('--INIT_XYZS', default=[0], type=list, help='Initial drone positions')
    parser.add_argument('--INIT_TARGETS', default=[0], type=list, help='Initial target positions')

    parser.add_argument('--skewed_plane_BVC', default=False, type=bool,
                        help='Select, whether the BVC planes should be skewed')
    parser.add_argument('--event_trigger', default=False, type=bool,
                        help='Select, whether the event trigger should be used for scheduling')
    parser.add_argument('--downwash_scaling_factor', default=2, type=int,
                        help='Scaling factor to account for the downwash')
    parser.add_argument('--downwash_scaling_factor_crit', default=3, type=int,
                        help='Scaling factor to account for the downwash')
    parser.add_argument('--use_qpsolvers', default=True, type=bool,
                        help='Select, whether qpsolver is used for trajectory planning')
    parser.add_argument('--alpha_1', default=10.0, type=bool,
                        help='Weight in event-trigger')
    parser.add_argument('--alpha_2', default=1.0, type=bool,
                        help='Weight in event-trigger')
    parser.add_argument('--alpha_3', default=1.0, type=bool,
                        help='Weight in event-trigger')
    ARGS = parser.parse_args()

    path = os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/" \
                                                        "dmpc_simulation_results_test_" + datetime.now().strftime(
        "%m_%d_%Y_%H_%M_%S")



    ARGS.path = path
    create_dir(path)

    ### Initialize the plotter #################################
    plotter = Plotter()

    # Initialize the simulations
    simulations = []

    # prepare the ARGS for simulations
    ARGS_array = []
    testbed = cuboid.Cuboid(np.array([0, 0, 1]), np.array([ARGS.testbed_size[0], 0, 0]),
                            np.array([0, ARGS.testbed_size[1], 0]),
                            np.array([0, 0, ARGS.testbed_size[2]]))

    initializer = initializer.Initializer(testbed, rng_seed=2)

    # If the loaded ARGS_array is sorted by num_drones, some information needs to be given in order to load them correctly
    num_drones_in_ARGS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    loaded_sims_per_drone_config = [0] * len(num_drones_in_ARGS)
    num_sims_per_drone_config_in_ARGS = 1000

    INITS_const = [[-1, 1, 1],
                   [1, 1, 1],
                   [-1, 0, 1],
                   [1, 0, 1],
                   [-1, -1, 1],
                   [1, -1, 1]]
    targets_const = [[1, 0, 1],
                   [-1, 1, 1],
                   [1, -1, 1],
                   [-1, 0, 1],
                   [1, 1, 1],
                   [-1, -1, 1]]
    use_inits_const = False

    load_data_path=None
    ARGS_loaded = None
    ARGS_sorted_by_num_drones = False  # select, whether the ARGS_array is sorted
    if load_data_path is not None:
        ARGS_loaded = p.load(open(load_data_path, "rb"))

    for i in range(0, ARGS.total_simulations):
        ARGS_for_simulation = copy.deepcopy(ARGS)
        num_sims_per_drone_config = int(ARGS.total_simulations / len(ARGS.num_drones))
        if num_sims_per_drone_config * len(ARGS.num_drones) != ARGS.total_simulations:
            raise ValueError('Total Simulations is not an integer multiple of the total number of drone configurations')

        if ARGS_loaded is not None and not use_inits_const and ARGS_sorted_by_num_drones:
            ARGS_for_simulation.num_drones = ARGS.num_drones[int(i / num_sims_per_drone_config)]
        else:
            ARGS_for_simulation.num_drones = ARGS.num_drones[i%len(ARGS.num_drones)]
        ARGS_for_simulation.network_message_loss = ARGS.network_message_loss[int(i / len(ARGS.num_drones)) %
                                                                             len(ARGS.network_message_loss)]

        if ARGS_loaded is None:
            ARGS_for_simulation.INIT_XYZS, ARGS_for_simulation.INIT_TARGETS = initializer.initialize(
                ARGS_for_simulation.drone_position_initialization_method, dist_to_wall=0.25,
                num_points=ARGS_for_simulation.num_drones,
                min_dist=ARGS.r_min, scaling_factor=ARGS.downwash_scaling_factor)
        elif not use_inits_const:
            if ARGS_sorted_by_num_drones:
                current_drone_config = ARGS_for_simulation.num_drones
                idx = np.where(num_drones_in_ARGS == current_drone_config)[0][0]
                ARGS_for_simulation.INIT_XYZS = ARGS_loaded[loaded_sims_per_drone_config[idx] + idx *
                                                            num_sims_per_drone_config_in_ARGS].INIT_XYZS
                ARGS_for_simulation.INIT_TARGETS = ARGS_loaded[loaded_sims_per_drone_config[idx] + idx *
                                                               num_sims_per_drone_config_in_ARGS].INIT_TARGETS
                loaded_sims_per_drone_config[idx] += 1
            else:
                print("Hello")
                ARGS_for_simulation.INIT_XYZS = ARGS_loaded[i].INIT_XYZS
                ARGS_for_simulation.INIT_TARGETS = ARGS_loaded[i].INIT_TARGETS
        else:
            ARGS_for_simulation.INIT_XYZS = INITS_const
            ARGS_for_simulation.INIT_TARGETS = targets_const


        print("Prepared simulation number " + str(i) + " with " + str(ARGS_for_simulation.num_drones) + " Drones.")
        ARGS_for_simulation.sim_id = i + 1
        ARGS_array.append(ARGS_for_simulation)

    with open(path + "/ARGS.pkl", 'wb') as out_file:
        pickle.dump(ARGS_array, out_file)

    #exit()
    # Run the simulations in parallel mode
    start = time.time()

    if ARGS.multiprocessing and ARGS.total_simulations > 0:
        max_threads = multiprocessing.cpu_count() - 2
        p = mp.Pool(processes=np.min((max_threads, ARGS.total_simulations)), maxtasksperchild=1)  #
        simulation_logger = [x for x in p.imap(parallel_simulation_wrapper, ARGS_array)]
        p.close()
        p.terminate()
        p.join()

    # run the simulation in batch mode
    else:
        simulations = []
        # initialize simulations
        for current_ARGS in ARGS_array:
            simulations.append(simulation.Simulation(current_ARGS))

        # run simulations
        simulation_logger = []
        for sim in simulations:
            simulation_logger.append(sim.run())

    # Plot the simulation sequentially
    if ARGS.plot:
        # plotter.plot('3D', path=path)
        plotter.plot('2D', path=path)

    print('Total simulation time: ' + str(round(time.time() - start, 2)) + 's.')

    # save the simulation result and ARGS
    # if ARGS.total_simulations != 0 and ARGS.log:
    #    save(simulation_logger, ARGS)

    if ARGS.event_trigger:
        title_addon = 'Event Trigger'
    else:
        title_addon = 'Round Robin'

    if ARGS.plot_batch_results:
        plotter.plot_batch_sim_results('av_transition_time', path=path, title_addon=title_addon)
        # plotter.plot_batch_sim_results('success_rate', path=path, title_addon='Event Trigger')
        # plotter.plot_batch_sim_results('dist_to_target_over_time', path=path)
        # plotter.plot_batch_sim_results('crashed', path=path)

    if not ARGS.log:
        shutil.rmtree(path)

    exit()
