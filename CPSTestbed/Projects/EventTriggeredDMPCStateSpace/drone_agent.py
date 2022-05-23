import math

from CPSTestbed.network import network as net
import CPSTestbed.Projects.EventTriggeredDMPCStateSpace.trajectory_generation.trajectory_generation as tg
from CPSTestbed.Projects.EventTriggeredDMPCStateSpace.trajectory_generation.interpolation import \
    PiecewiseBernsteinPolynomial
from CPSTestbed.Projects.EventTriggeredDMPCStateSpace.trajectory_generation.statespace_model import TripleIntegrator
import copy
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class TrajectoryMessageContent:
    id: int  # the corresponding id for the coefficients
    coefficients: any


class ComputationAgent(net.Agent):
    """this class represents a computation agent inside the network.
    Every agent is allowed to send in multiple slot groups. This is needed if e.g. the agent should send his sensor
    measurements but also if he is in error mode.

    Methods
    -------
    get_prio(slot_group_id):
        returns current priority
    get_message(slot_group_id):
        returns message
    send_message(message):
        send a message to this agent
    round_finished():
        callback for the network to signal that a round has finished
    """

    def __init__(self, ID, slot_group_planned_trajectory_id, init_positions, target_positions, agents_ids,
                 communication_delta_t,
                 trajectory_generator_options, prediction_horizon, num_computing_agents, offset=0, use_event_trigger=False,
                 alpha_1=10, alpha_2=1, alpha_3=1):
        """

        Parameters
        ----------
            ID:
                identification number of agent
            slot_group_planned_trajectory_id:
                ID of the slot group which is used
            init_positions:
                3D position of agents in a hash_map id of agent is key
            num_agents: int
                total number of agents in the system
            trajectory_generator_options: tg.TrajectoryGeneratorOptions
                options for trajectory generator
            agents_ids: numpy.array, shape (num_agents,)
                ids of all agents in the system
            communication_delta_t: float
                time difference between two communication rounds
            prediction_horizon: int
                prediction horizon of optimizer
            order_interpolation: int
                order of interpolation

        """
        super().__init__(ID, [slot_group_planned_trajectory_id])
        self.__slot_group_planned_trajectory_id = slot_group_planned_trajectory_id
        self.__target_positions = copy.deepcopy(target_positions)
        self.__agent_state = np.array(
            [np.hstack((copy.deepcopy(init_positions[id]), np.zeros(2 * init_positions[id].shape[0]))).ravel()
             for id in agents_ids])

        self.__agents_ids = agents_ids
        self.__communication_delta_t = communication_delta_t

        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2
        self.__alpha_3 = alpha_3

        self.__current_time = 0
        self.__prediction_horizon = prediction_horizon

        # communication timepoints start with the time of the next communication. (we have a delay of one due to the
        # communication) * 2, because the current position cannot be influenced anymore
        self.__communication_timepoints = \
            np.linspace(self.__communication_delta_t * 2,
                        self.__communication_delta_t * 2 + communication_delta_t * prediction_horizon,
                        prediction_horizon)
        self.__breakpoints = \
            np.linspace(0, communication_delta_t * prediction_horizon, prediction_horizon * int(communication_delta_t
                                                                                                / trajectory_generator_options.optimization_variable_sample_time) + 1)

        # planned for the next timestep

        self.__agents_coefficients = {}
        self.__agents_coefficients_calculated = {}
        self.__agents_starting_times = {}  # times the corresponding coefficents start.
        for id in agents_ids:
            self.__agents_coefficients[id] = \
                tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((prediction_horizon * int(communication_delta_t /
                                                                             trajectory_generator_options.optimization_variable_sample_time),
                                                    init_positions[id].shape[0])))

            self.__agents_coefficients_calculated[id] = \
                tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((prediction_horizon * int(communication_delta_t /
                                                                             trajectory_generator_options.optimization_variable_sample_time),
                                                    init_positions[id].shape[0])))
            self.__agents_starting_times[id] = 0

        self.__trajectory_interpolation = TripleIntegrator(
            breakpoints=self.__breakpoints,
            dimension_trajectory=3,
            num_states=9,
            sampling_time=trajectory_generator_options.optimization_variable_sample_time)

        self.__trajectory_generator = tg.TrajectoryGenerator(
            options=trajectory_generator_options,
            trajectory_interpolation=self.__trajectory_interpolation)

        self.__number_rounds = 0
        self.__options = trajectory_generator_options

        self.__current_agent = offset % len(agents_ids)

        self.__num_agents = len(agents_ids)
        self.__num_computing_agents = num_computing_agents
        self.__agent_prios = np.zeros((self.__num_agents,))

        self.__use_event_trigger = use_event_trigger

    def get_prio(self, slot_group_id):
        """returns the priority for the schedulder, because the system is not event triggered, it just returns zero

        Returns
        -------
            prio:
                priority
            slot_group_id:
                the slot group id the prio should be calculated to
        """
        return 1

    def get_message(self, slot_group_id):
        """returns the current message that should be send.
        If the agent is not in the slot group, it will return None

        Parameters
        ----------
            slot_group_id:
                the slot group id the message should belong to

        Returns
        -------
            message: Message
                Message
        """
        if slot_group_id == self.__slot_group_planned_trajectory_id:
            # copy array because the agent might change this value during the network simulation and otherwise this new
            # value will be transmitted
            planned_trajectory_future = copy.deepcopy(
                self.__agents_coefficients_calculated[self.__agents_ids[self.__current_agent]])
            return net.Message(self.ID, slot_group_id, TrajectoryMessageContent(self.__agents_ids[self.__current_agent],
                                                                                planned_trajectory_future))

    def send_message(self, message):
        """send message to agent.

        Parameters
        ----------
            message: Message
                message to send.
        """
        # if the message is from leader agent set new reference point
        if message.slot_group_id == self.__slot_group_planned_trajectory_id:
            self.__agents_coefficients[message.content.id] = message.content.coefficients
            self.__agents_starting_times[message.content.id] = self.__current_time

    def round_finished(self):
        """this function has to be called at the end of the round to tell the agent that the communication round is
        finished"""
        start_time = time.time()

        # until the agents have not send their real trajectory data
        if self.__number_rounds <= 1:
            self.__number_rounds = self.__number_rounds + 1
            return

        for i in range(0, self.__num_agents):
            # calculate current state of all agents
            id = self.__agents_ids[i]
            self.__agent_state[id] = self.__trajectory_interpolation.interpolate(
                self.__current_time - self.__agents_starting_times[id] + self.__communication_delta_t,
                self.__agents_coefficients[id],
                x0=self.__agent_state[id],
                integration_start=self.__current_time - self.__agents_starting_times[id])

        # choose next agent
        if not self.__use_event_trigger:
            self.__current_agent = (self.__current_agent + 1) % self.__num_agents
            current_id = self.__agents_ids[self.__current_agent]
        else:
            self.__current_agent = self.select_next_agent()
            current_id = self.__agents_ids[self.__current_agent]

        # calculate obstacle trajectories
        other_agent_trajectories = np.zeros((self.__num_agents - 1,
                                             len(self.__options.collision_constraint_sample_points), 3))
        other_targets = np.zeros((self.__num_agents - 1, 3))
        other_ids = np.zeros((self.__num_agents - 1,))
        j = 0
        for i in range(0, self.__num_agents):
            if i == self.__current_agent:
                continue
            id = self.__agents_ids[i]
            # calculate trajectories of other agents
            other_agent_trajectories[j, :, :] = self.__trajectory_interpolation.interpolate_vector(
                self.__current_time - self.__agents_starting_times[id] + self.__communication_delta_t +
                self.__options.collision_constraint_sample_points,
                self.__agents_coefficients[
                    id], x0=self.__agent_state[id],
                derivative_order=0,
                integration_start=self.__current_time - self.__agents_starting_times[id] + self.__communication_delta_t)

            other_targets[j, :] = self.target_positions[id]
            other_ids[j] = id
            j = j + 1

        x_vector = self.__current_time - self.__agents_starting_times[
            current_id] + self.__communication_delta_t + self.__options.collision_constraint_sample_points
        # current trajectory at anti collision sample points, used for anti collision constraints
        current_planned_trajectory = self.__trajectory_interpolation.interpolate_vector(
            x_vector, self.__agents_coefficients[current_id], derivative_order=0, x0=self.__agent_state[current_id],
            integration_start=self.__current_time - self.__agents_starting_times[
                current_id] + self.__communication_delta_t)

        delay_timesteps = (self.__current_time - self.__agents_starting_times[
            # calulate number of timesteps, the alternative trajectory need to be delayed
            current_id] + self.__communication_delta_t) / self.__options.optimization_variable_sample_time
        delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies
        if self.__agents_coefficients[current_id].valid:
            previous_solution_shifted = self.__agents_coefficients[current_id].coefficients
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])
        else:
            previous_solution_shifted = self.__agents_coefficients[current_id].alternative_trajectory
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])

        self.__agents_coefficients_calculated[current_id] = self.__trajectory_generator.calculate_trajectory(
            current_state=self.__agent_state[current_id],
            target_position=self.__target_positions[current_id],
            planned_trajectory=current_planned_trajectory,
            other_trajectories=other_agent_trajectories,
            previous_solution=previous_solution_shifted,
            drone_id=current_id,
            timestep=self.__current_time,
            other_targets=other_targets,
            other_ids=other_ids
        )
        # start time of the newly calculated trajectory.
        self.__number_rounds = self.__number_rounds + 1
        self.__current_time = self.__current_time + self.__communication_delta_t

    def select_next_agent(self):
        """ selects the next agent to update its trajectory
        Returns:
            next_agent: int
                number of the agent to update the trajectory for"""

        prios = self.calc_prio()
        slot_no = self.ID - self.__num_agents  # better way?
        next_agent_prios = sorted(prios)[-self.__num_computing_agents:]
        next_agents = []
        for next_agent_prio in next_agent_prios:
            next_agent = prios.tolist().index(next_agent_prio)
            prios[next_agent] = -999  # dirty fix to prevent that the same agent is chosen by the computing nodes if multiple have the same priority
            next_agents.append(next_agent)
        return next_agents[slot_no]

    def calc_prio(self):
        """ calculates the prio for each agent """
        max_time = self.__prediction_horizon * self.__communication_delta_t
        max_dist = np.linalg.norm(self.__options.max_position - self.__options.min_position)
        cone_angle = 60.0 * math.pi / 180

        prios = np.zeros((self.__num_agents, ))
        for i in range(0, self.__num_agents):
            own_id = self.__agents_ids[i]

            d_target = self.target_positions[own_id] - self.agent_positions[own_id]
            dist_to_target = np.linalg.norm(d_target)

            d_target = d_target / dist_to_target  # normalized target vector

            for j in range(0, self.__num_agents):
                if i == j:
                    continue
                obstacle_id = self.__agents_ids[j]

                d_obst = self.agent_positions[obstacle_id] - self.agent_positions[own_id]
                dist_to_obst = np.linalg.norm(d_obst)
                d_obst = d_obst / dist_to_obst  # normalized obstacle vector

                scaling = 1 * max((0, dist_to_target - dist_to_obst))  # closer drones have a higher priority 0.05
                angle = np.dot(d_target, d_obst)
                angle = min(angle, 1)
                angle = max(angle, -1)  # just in case of numerical problems

                if math.acos(angle) <= cone_angle:
                    prios[i] -= scaling * angle
                
            prios[i] /= (1.0 * self.__num_agents * max_dist)  # normalize prio

            prios[i] *= self.__alpha_3
            prios[i] += self.__alpha_2 * (self.__current_time - self.__agents_starting_times[own_id]) / max_time #0.1
            prios[i] += self.__alpha_1 * dist_to_target / max_dist

        return prios

    def print(self, text):
        print("[" + str(self.ID) + "]: " + str(text))

    def round_started(self):
        pass

    @property
    def target_positions(self):
        return self.__target_positions

    @target_positions.setter
    def target_positions(self, target_positions):
        self.__target_positions = target_positions

    @property
    def agent_positions(self):
        return self.__agent_state[:, :3]


class RemoteDroneAgent(net.Agent):
    """this class represents a drone agent inside the network.
    Every agent is allowed to send in multiple slot groups. This is needed if e.g. the agent should send his sensor
    measurements but also if he is in error mode. This agent receives a new reference trajectory and follows it.

    Methods
    -------
    get_prio(slot_group_id):
        returns current priority
    get_message(slot_group_id):
        returns message
    send_message(message):
        send a message to this agent
    round_finished():
        callback for the network to signal that a round has finished
    """

    def __init__(self, ID, slot_group_planned_trajectory_id, init_position, target_position, communication_delta_t,
                 trajectory_generator_options, prediction_horizon, order_interpolation=4):
        """

        Parameters
        ----------
            ID:
                identification number of agent
            slot_group_planned_trajectory_id:
                ID of the slot group which is used
            init_position: np.array
                3D position of agent
            trajectory_generator_options: tg.TrajectoryGeneratorOptions
                options for trajectory generator
            communication_delta_t: float
                time difference between two communication rounds
            prediction_horizon: int
                prediction horizon of optimizer
            order_interpolation: int
                order of interpolation

        """
        super().__init__(ID, [])
        self.__slot_group_planned_trajectory_id = slot_group_planned_trajectory_id
        self.__dim = init_position.shape[0]
        self.__traj_state = np.hstack((copy.deepcopy(init_position), np.zeros(
            2 * init_position.shape[0]))).ravel()
        self.__position = copy.deepcopy(init_position)
        self.__target_position = np.copy(target_position)
        self.__next_pos_trajectory = np.copy(target_position)
        self.__init_state = self.__traj_state
        self.__transition_time = None
        self.__target_reached = False
        self.__crit_distance_to_target = 0.2  # m
        self.__communication_delta_t = communication_delta_t
        self.__crashed = False

        self.__current_time = 0
        self.__planned_trajectory_start_time = -1
        self.__prediction_horizon = prediction_horizon

        # communication timepoints start with the time of the next communication. (we have a delay of one due to the
        # communication) * 2, because the current position cannot be influenced anymore
        self.__communication_timepoints = \
            np.linspace(self.__communication_delta_t * 2,
                        self.__communication_delta_t + communication_delta_t * prediction_horizon,
                        prediction_horizon)

        # breakpoints of optimization variable
        self.__breakpoints = \
            np.linspace(0, communication_delta_t * prediction_horizon, prediction_horizon * int(
                communication_delta_t / trajectory_generator_options.optimization_variable_sample_time) + 1)

        # planned for the next timestep
        self.__planned_trajectory_coefficients = tg.TrajectoryCoefficients(None, False,
                                                                           np.zeros((
                                                                               int(prediction_horizon * communication_delta_t /
                                                                                   trajectory_generator_options.optimization_variable_sample_time),
                                                                               self.__dim)))

        self.__trajectory_interpolation = TripleIntegrator(
            breakpoints=self.__breakpoints,
            dimension_trajectory=3,
            num_states=9,
            sampling_time=trajectory_generator_options.optimization_variable_sample_time)

        self.__number_rounds = 0
        self.__options = trajectory_generator_options

    def get_prio(self, slot_group_id):
        """returns the priority for the scheulder, because the system is not event triggered, it just returns zero

        Returns
        -------
            prio:
                priority
            slot_group_id:
                the slot group id the prio should be calculated to
        """
        return 0

    def send_message(self, message):
        """send message to agent.

        Parameters
        ----------
            message: Message
                message to send.
        """
        # if the message is from leader agent set new reference point
        if message.slot_group_id == self.__slot_group_planned_trajectory_id:
            if message.content.id == self.ID:
                self.__planned_trajectory_coefficients = copy.deepcopy(message.content.coefficients)
                self.__planned_trajectory_start_time = self.__current_time

    def round_finished(self):
        """this function has to be called at the end of the round to tell the agent that the communication round is
        finished"""
        pass

    def print(self, text):
        print("[" + str(self.ID) + "]: " + str(text))

    def round_started(self):
        pass

    def next_planned_state(self, delta_t):
        """
        Parameters
        ----------
            delta_t: float
                time between the last and this call of the function or time between the call of this function
                and the last call of round_started()
        Returns
        -------
            next_planned_state: int
                state for next position controller
        """
        if self.__crashed:
            self.__traj_state[self.__dim] = 0
            return self.__traj_state
        if self.__planned_trajectory_start_time >= 0:
            self.__traj_state = self.__trajectory_interpolation.interpolate(
                self.__current_time - self.__planned_trajectory_start_time + delta_t,
                self.__planned_trajectory_coefficients, integration_start=self.__current_time -
                                                                          self.__planned_trajectory_start_time,
                x0=self.__traj_state)

            self.__current_time = self.__current_time + delta_t

            return self.__traj_state
        return self.__init_state

    @property
    def position(self):
        """returns current measured positions

        Returns
        -------
        positions: np.array
            3D position of drone
        """
        return self.__position

    @position.setter
    def position(self, pos):
        """sets current measured position of drone"""
        self.__position = copy.deepcopy(pos)

    @property
    def x_ref(self):
        """
        returns current reference position

        Returns
        -------
        x_ref: np.array
            3D reference position of agent
        """
        return self.__x_ref

    @x_ref.setter
    def x_ref(self, new_x_ref):
        """sets new reference value.
        If the id of the drone is not the id of the leader. This function does nothing."""
        if self.ID == 0:
            self.__x_ref = new_x_ref

    @property
    def target_position(self):
        """
        returns target position
        """
        return self.__target_position

    @target_position.setter
    def target_position(self, new_target_position):
        self.__target_position = new_target_position

    @property
    def target_reached(self):
        return self.__target_reached

    @target_reached.setter
    def target_reached(self, new_target_reached_state):
        self.__target_reached = new_target_reached_state

    @property
    def transition_time(self):
        return self.__transition_time

    @transition_time.setter
    def transition_time(self, new_transition_time):
        """ sets the transition time"""
        self.__transition_time = new_transition_time

    @property
    def current_time(self):
        """ returns the current time of the drone"""
        return self.__current_time

    @property
    def traj_state(self):
        """ returns current state of the trajectory"""
        return self.__traj_state

    @property
    def crashed(self):
        """ returns the crashed status of the drone"""
        return self.__crashed

    @crashed.setter
    def crashed(self, new_crashed_status):
        """ sets the crashed status of the drone"""
        self.__crashed = new_crashed_status
