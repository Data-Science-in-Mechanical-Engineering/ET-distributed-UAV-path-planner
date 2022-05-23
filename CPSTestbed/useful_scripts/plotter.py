import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from cycler import cycler
import pickle


class Plotter():
    """ This class represents a plotter to create individual plots of simulations"""

    def __init__(self):
        """
        Parameters: path is static
        """
        self.__path_single = "../../../gym-pybullet-drones/files/logs/*.npy"
        self.__path_batch = "../../batch_simulation_results/*.pkl"
        self.__states = None
        self.__timestamps = None
        self.__controls = None
        self.__targets = None

    def load(self, path):
        """ Method to load the data for plotting. If a logger is provided, the data from the logger is used. Otherwise,
        the most recent saved data is used.

        Parameters:
            logger: logger containing simulation results"""

        if '/*.pkl' not in path:
            path = path + "/*.pkl"
        files = glob.glob(path)
        simulation_results = []
        for file in files:
            with open(file, 'rb') as currentFile:
                current_sim_res = pickle.load(currentFile)
            if type(current_sim_res).__name__ == 'CustomLogger':
                simulation_results.append(current_sim_res)
            if type(current_sim_res) is list:
                ARGS_array = current_sim_res

        return simulation_results, ARGS_array

    def plot(self, plot_type, logger=None, drone_ids=None, path=None):
        """ plots the drones trajectories in 2D and 3D"""

        if logger is not None:
            simulation_results = [logger]

        elif path is not None:
            simulation_results, ARGS = self.load(path)

        else:
            print('Nothing to plot, aborting')
            return

        for result in simulation_results:
            states = result.states
            controls = result.controls
            timestamps = result.timestamps
            targets = result.targets

            if drone_ids is None:
                drone_ids, _, _ = states.shape
                drone_ids = range(drone_ids)

            if type(drone_ids) == int:
                drone_ids = [drone_ids]

            if plot_type == '2D':
                plt.rc('axes', prop_cycle=cycler('color', ['r', 'g', 'b', 'y', 'm', 'c']))
                fig, ax = plt.subplots(1, 1)
                ax.set_title('2D Position of drones')
                ax.set_ylabel('y Position [m]')
                ax.set_xlabel('x Position [m]')

                ax.grid()
                for i in drone_ids:
                    x_pos = states[i, 0, :]
                    y_pos = states[i, 1, :]
                    target_x = targets[i, 0, :]
                    target_y = targets[i, 1, :]
                    ax.plot(x_pos, y_pos, label='Drone ' + str(i))
                    ax.scatter(target_x, target_y, c='k')

                ax.set_aspect('equal', adjustable='datalim')
                plt.show()

            if plot_type == '3D':
                plt.rc('axes', prop_cycle=cycler('color', ['r', 'g', 'b', 'y', 'm', 'c']))
                fig = plt.figure(figsize=(10, 10))

                ax = fig.add_subplot(projection='3d')
                ax.patch.set_facecolor('white')
                ax.set_title('3D Position of drones')
                ax.set_xlabel('x Position [m]')
                ax.set_ylabel('y Position [m]')
                ax.set_zlabel('z Position [m]')

                for i in drone_ids:
                    x_pos = states[i, 0, :]
                    y_pos = states[i, 1, :]
                    z_pos = states[i, 2, :]
                    target_x = targets[i, 0, :]
                    target_y = targets[i, 1, :]
                    target_z = targets[i, 2, :]
                    ax.scatter(x_pos, y_pos, z_pos, label='Drone ' + str(i), s=1)
                    ax.scatter(target_x, target_y, target_z, c='k')

                plt.show()

    def plot_batch_sim_results(self, plot_type, path=None, simulation_results=None, title_addon=""):
        """ plots the batch results of the simulation run.

        Parameters:
            plot_type: string
                specify the type of plot
            path: string
                path to the parent folder of the simulation run. Use this, if every simulation run is stored in an individual file
            simulation_results: list of simulation_result Objects
                simulation results from a simulation run stored as list
            title_addon: string
                addon to the title of the plots
            """
        if simulation_results is None and path is None:  # to keep functionality for previously stored simulation results
            files = glob.glob(self.__path_batch)
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'rb') as file:
                simulation_results, ARGS = pickle.load(file)
        elif path is not None:
            simulation_results, ARGS_array = self.load(path)

        if len(title_addon) > 0:
            title_addon = ', ' + title_addon

        if plot_type == 'av_transition_time':
            transition_times = {}  # dict containing transition times sorted by the number of drones

            # fill the transition_times dict
            for result in simulation_results:

                result.transition_times = np.array([np.NaN if result.transition_times[i] < 1 else result.transition_times[i] for i in range(len(result.transition_times))])
                if result.num_drones in transition_times.keys():
                    transition_times[result.num_drones] = np.hstack((transition_times[result.num_drones],
                                                                     result.transition_times))
                else:
                    transition_times[result.num_drones] = result.transition_times

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(
                'Time to reach the target' + title_addon + '\n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Number of drones')
            ax.set_ylabel('Time [s]')
            ax.grid()

            data = [transition_times[key] for key in transition_times.keys()]
            data = [dat[~np.isnan(dat)] for dat in data]
            x = [key for key in transition_times.keys()]

            ax.boxplot(data, showfliers=True)

            ax.set_xticklabels(x)

            plt.show()

        if plot_type == 'success_rate':
            # initialize dict that saves the successful status of each drone for each simulation
            success = {}

            for result in simulation_results:
                if result.num_drones in success.keys():
                    success[result.num_drones] = np.hstack((success[result.num_drones],
                                                            result.successful))
                else:
                    success[result.num_drones] = result.successful

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Success rate' + title_addon + '\n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Number of drones')
            ax.set_ylabel('Success rate [%]')
            ax.grid()

            # for each drone configuration, calculate the average transition time
            for key in success.keys():
                y = np.nanmean(success[key]) * 100.0  # calcualte average success rate in percent
                x = key
                ax.scatter(x, y, c='b')

            plt.show()

        if plot_type == 'dist_to_target_over_time':
            dist_to_target = {}
            timestamps = {}
            for k in range(len(simulation_results)):
                result = simulation_results[k]
                ARGS = ARGS_array[k]

                timestamps[result.num_drones] = np.linspace(0, ARGS.duration_sec,
                                                            ARGS.duration_sec * ARGS.control_freq_hz + 1)

                # calc dist to target over time of each drone and store it
                for i in range(result.num_drones):
                    pos = result.states[i, 0:3, :]
                    target = result.targets[i]
                    crashed = result.crashed[i]

                    # don't use this drone if it crashed
                    if hasattr(crashed, "__len__"):
                        if any(crashed):
                            continue

                    dist = np.full(len(timestamps[result.num_drones]), np.NaN)
                    for j in range(len(result.timestamps[i, :])):
                        dist[j] = np.linalg.norm(pos[:, j] - np.transpose(target))

                    if result.num_drones in dist_to_target.keys():
                        dist_to_target[result.num_drones] = np.vstack((dist_to_target[result.num_drones],
                                                                       dist))
                    else:
                        dist_to_target[result.num_drones] = dist

            plt.rc('axes', prop_cycle=cycler('color', ['r', 'g', 'b', 'y', 'm', 'c']))
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(
                'Average Distance To Target' + title_addon + ' \n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Distance To Target [m]')
            ax.grid()

            for key in dist_to_target.keys():
                y = [np.nanmean(dist_to_target[key][:, i]) for i in range(len(timestamps[key]))]
                x = timestamps[key]
                # plt.errorbar(x, y, err, linestyle='None', fmt='o', c='b', capsize=5)
                ax.plot(x, y, label=str(key) + ' Drones')

            ax.legend()
            plt.show()

        if plot_type == 'crashed':
            # initialize dict that saves the crashed status of each drone for each simulation
            crashed = {}

            for result in simulation_results:
                if hasattr(result.crashed, "__len__"):
                    drone_crashed = result.crashed[:, -1]
                else:
                    drone_crashed = result.crashed

                if result.num_drones in crashed.keys():

                    crashed[result.num_drones] = np.hstack((crashed[result.num_drones],
                                                            drone_crashed))
                else:
                    crashed[result.num_drones] = drone_crashed

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Crash rate' + title_addon + ' \n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Number of drones')
            ax.set_ylabel('Crash rate [%]')
            ax.grid()

            # for each drone configuration, calculate the average transition time
            for key in crashed.keys():
                y = np.nanmean(crashed[key]) * 100.0  # calcualte average success rate in percent
                x = key
                # plt.errorbar(x, y, err, linestyle='None', fmt='o', c='b', capsize=5)
                ax.scatter(x, y, c='b')

            plt.show()

        if plot_type == 'success_vs_message_loss':
            # initialize dict that saves the success status of each drone for each simulation

            success = {}
            for result in simulation_results:
                if result.ARGS.network_message_loss in success.keys():
                    success[result.ARGS.network_message_loss] = np.hstack((success[result.ARGS.network_message_loss],
                                                                           result.successful))
                else:
                    success[result.ARGS.network_message_loss] = result.successful

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Average success rate' + title_addon + ' \n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Message loss')
            ax.set_ylabel('Success rate [%]')
            ax.grid()

            # for each drone configuration, calculate the average transition time
            for key in success.keys():
                y = np.nanmean(success[key]) * 100.0  # calcualte average success rate in percent
                x = key
                # plt.errorbar(x, y, err, linestyle='None', fmt='o', c='b', capsize=5)
                ax.scatter(x, y, c='b')

            plt.show()

        if plot_type == 'crashed_vs_message_loss':
            # initialize dict that saves the crashed status of each drone for each simulation
            crashed = {}

            for result in simulation_results:
                if hasattr(result.crashed, "__len__"):
                    drone_crashed = result.crashed[:, -1]
                else:
                    drone_crashed = result.crashed
                if result.ARGS.network_message_loss in crashed.keys():
                    crashed[result.ARGS.network_message_loss] = np.hstack((crashed[result.ARGS.network_message_loss],
                                                                           drone_crashed))
                else:
                    crashed[result.ARGS.network_message_loss] = drone_crashed

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Crash rate' + title_addon + ' \n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Message loss [%]')
            ax.set_ylabel('Crash rate [%]')
            ax.grid()

            # for each drone configuration, calculate the average transition time
            for key in crashed.keys():
                y = np.nanmean(crashed[key]) * 100.0  # calcualte average success rate in percent
                x = key * 100.0
                # plt.errorbar(x, y, err, linestyle='None', fmt='o', c='b', capsize=5)
                ax.scatter(x, y, c='b')

            plt.show()

        if plot_type == 'crashed_vs_message_loss_drone_count':
            # initialize dict that saves the crashed status of each drone for each simulation
            crashed = {}

            for result in simulation_results:
                # only take final crashed status
                drone_crashed = result.crashed[:, -1]

                # create dict that sorts the crashed status by number of drones and message loss
                if result.num_drones in crashed.keys():
                    if result.ARGS.network_message_loss in crashed[result.num_drones].keys():
                        crashed[result.num_drones][result.ARGS.network_message_loss] = np.hstack((
                            crashed[result.num_drones][result.ARGS.network_message_loss], drone_crashed))
                    else:
                        crashed[result.num_drones][result.ARGS.network_message_loss] = drone_crashed
                else:
                    crashed[result.num_drones] = {}
                    crashed[result.num_drones][result.ARGS.network_message_loss] = drone_crashed

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Average crash rate' + title_addon + ' \n ' + str(len(simulation_results)) + ' Simulations')
            ax.set_xlabel('Message loss [%]')
            ax.set_ylabel('Crash rate [%]')
            ax.grid()

            for drone_count in crashed.keys():
                y = []
                x = []
                for message_loss in crashed[drone_count].keys():
                    y.append(np.nanmean(
                        crashed[drone_count][message_loss]) * 100.0)  # calcualte average success rate in percent
                    x.append(message_loss * 100.0)
                    # plt.errorbar(x, y, err, linestyle='None', fmt='o', c='b', capsize=5)
                ax.plot(x, y, label=str(drone_count) + " Drones")

            ax.legend()
            plt.show()

    @property
    def path_single(self):
        """ Method that returns the path to the log files"""
        return self.__path_single

    @path_single.setter
    def path(self, new_path):
        """ Method to set the path variable"""
        self.__path_single = new_path

    @property
    def path_bath(self):
        """ Method to return the path to the batch simulation results"""
        return self.__path_batch

    @path_bath.setter
    def path_batch(self, new_path):
        """ Method to set a new path for the batch simulation results"""
        self.__path_batch = new_path

    @property
    def timestamps(self):
        """ Method that returns the timestamps"""
        return self.__timestamps

    @timestamps.setter
    def timestamps(self, new_timestamps):
        """ Method to set the timestamps"""

        self.__timestamps = new_timestamps

    @property
    def states(self):
        """ Method that returns the states"""
        return self.__states

    @states.setter
    def states(self, new_states):
        """ Method that sets the states"""

        self.__states = new_states

    @property
    def controls(self):
        """ Method that returns the controls"""

        return self.__controls

    @controls.setter
    def controls(self, new_controls):
        """ Method that sets new controls"""

        self.__controls = new_controls
