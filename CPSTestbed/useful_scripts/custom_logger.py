import sys

sys.path.append("../../gym-pybullet-drones")

import numpy as np
import os

from gym_pybullet_drones.utils.Logger import Logger


class CustomLogger(Logger):
    """
    Class for logging of custom data
    """

    def __init__(self,
                 ARGS,
                 logging_freq_hz: int,
                 path = " ",
                 num_drones: int = 1,
                 duration_sec: int = 0,
                 ):
        """
        Constructor
        """

        super().__init__(logging_freq_hz=logging_freq_hz,
                         num_drones=num_drones,
                         duration_sec=duration_sec)

        self.__path = path
        # initialize target for each drone (static)
        self.targets = np.zeros((num_drones, 3, 1))
        # initialize successful flag for each drone, should be set if drone is at target at the end of the simulation
        self.successful = np.zeros(num_drones)
        # initialize required time to reach the target for each drone
        self.transition_times = np.zeros(num_drones)
        # initialize travelled distance
        self.distance_travelled = np.zeros(num_drones)
        # store number of drones
        self.num_drones = num_drones
        # store ARGS of this simulation
        self.ARGS = ARGS

        self.__positions = {}
        self.__control_positions = {}
        self.__times = {}
        self.crashed = {}

    def log_static(self,
                   drone: int,
                   successful: bool,
                   transition_time: any,
                   distance_travelled: float,
                   target, crashed):
        """Logs entries of simulation result

        Parameters:
            drone: int
                ID of the drone associated to the log entry
            successful: bool
                flag indicating if the drone reached it's target
            transition_time: float
                float containing the time needed to reach the target
            distance_travelled: float
                float containing the total distance travelled to reach the target
            target: np.array
                position of target
            crashed: bool
                flag if drone has crashed"""

        if drone < 0 or drone >= self.NUM_DRONES:
            print("[ERROR] in CustomLogger.log(), invalid data")

        # logs successful flag
        self.successful[drone] = successful
        # logs transition_times
        self.transition_times[drone] = transition_time
        # logs travelled distance
        self.distance_travelled[drone] = distance_travelled
        # log targets
        self.targets[drone] = np.transpose([target])

        self.crashed[drone] = crashed

    def log_position(self, ID, time, position, control_position):
        if ID not in self.__positions.keys():
            self.__positions[ID] = []
            self.__control_positions[ID] = []
            self.__times[ID] = []

        self.__positions[ID].append(position)
        self.__control_positions.append(control_position)
        self.__times[ID].append(time)

    def save_log(self):
        with open(os.path.join(self.__path, "Metadata.txt", 'w')) as f:
            for successful in self.successful:
                f.write(str(successful) + " ")
            f.write("\n")

            for transition_time in self.transition_times:
                f.write(str(transition_time) + " ")
            f.write("\n")

            for distance_travelled in self.distance_travelled:
                f.write(str(distance_travelled) + " ")
            f.write("\n")

            for targets in self.targets:
                f.write(str(targets) + " ")
            f.write("\n")

        for ID in self.__positions:
            with open(os.path.join(self.__path, "Position" + str() + ".txt", 'w')) as f:
                for i in range(len(self.__positions[ID])):
                    f.write(str(self.__times[ID][i]) + " " + str(self.__positions[ID][i][0])
                            + " " + str(self.__positions[ID][i][1])
                            + " " + str(self.__positions[ID][i][2]) + "\n")

        for ID in self.__control_positions:
            with open(os.path.join(self.__path, "ControlPosition" + str() + ".txt", 'w')) as f:
                for i in range(len(self.__control_positions[ID])):
                    f.write(str(self.__times[ID][i]) + " " + str(self.__control_positions[ID][i][0])
                            + " " + str(self.__control_positions[ID][i][1])
                            + " " + str(self.__control_positions[ID][i][2]) + "\n")


