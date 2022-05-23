import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import bisect

# constants
NORMAL = 0
ALL_TO_ALL = 1


class WrongArgumentError(ValueError):
    pass


class WrongSlotGroupError(ValueError):
    pass


class SlotGroup:
    """defines a slotgroup
    A slotgroup is a group of slots of the same type. The idea is that some slots are used for some type of data only and
    have a specific behaviour."""
    """
    defines a slotgroup
    A slotgroup is a group of slots of the same type. The idea is that some slots are used for some type of data only and
    have a specific behaviour.

    Attributes
    ----------
    id : int
        id of the slot group
    schedule : array
        array of agents scheduled for communication

    Methods
    -------
    clear_schedule():
        clears the schedule
        
    add_application_for_schedule():
        ask for a schedule for an agent.
        
    
    """

    def __init__(self, group_id, schedule_by_prio, num_slots):
        """constructor

        Parameters
        ----------
            group_id:
                identification number of the group. This number has to be unique over the whole network
            schedule_by_prio: boolean
                if false the slots are asigned in a all-to-all fashion (meaning during one round every agent can send)
                if true the slots are scheduled to the agents with the highest priorities
            num_slots:
                maximum numbers of slots available per round
            message_loss:
                probability of message loss
        """
        self.__group_id = group_id
        self.__schedule_by_prio = schedule_by_prio
        self.__num_slots = num_slots

        self.__slots_scheduled = []

        self.__agents_scheduled = []

        self.__application_prios = []

    @property
    def id(self):
        """
        Returns
        -------
            id:
                id of the group
        """
        return self.__group_id

    def clear_schedule(self):
        """clears the schedule
        This method has to be called after every round to make sure the schedule is cleared"""
        self.__agents_scheduled = []
        self.__application_prios = []

    def add_application_for_schedule(self, agent):
        """ask for a schedule for an agent.
        If schedule_by_prio in the constructor is set to False the agent will get a place.
        If schedule_by_prio in the constructor is set to True the agent will get a place if its priority is in the top
        __num_slots of prios of the other agents who applied.
        Parameters
        ----------
            agent:
                Agent to ask for

         Raises
        ------
            OverflowError:
                when every slot is already scheduled to an agent (if schedule_by_prio in the constructor is set to False)
        """
        if self.__schedule_by_prio:
            prio = agent.get_prio(self.__group_id)
            ind = bisect.bisect(self.__application_prios, prio)

            self.__application_prios.insert(ind, prio)
            self.__agents_scheduled.insert(ind, agent)

            if len(self.__application_prios) > self.__num_slots:
                self.__application_prios = self.__application_prios[1:]  # delete the worst element
                self.__agents_scheduled = self.__agents_scheduled[1:]
        else:
            if len(self.__agents_scheduled) >= self.__num_slots:
                raise OverflowError("Too many agents assigned to one slot")

            self.__agents_scheduled.append(agent)

    @property
    def schedule(self):
        """returns schedule"""
        return self.__agents_scheduled

    @property
    def num_slots(self):
        """returns the number of slots"""
        return self.__num_slots


class Network:
    """this class represents a multihop network with our approach.

    Methods
    -------
    add_agent(agent):
        adds agent to network.
    remove_agent(agent):
        removes agent from network
    add_slot_group(self, slot_group):
        adds slot group to network.
    step():
        performs one communication round.
    """

    def __init__(self, agents=None, message_loss=0):
        """constructor of class network

        Parameters
        ----------
            agents:
                list of agents that are connected over the network
        """

        if agents is None:
            self.__agents = []
        else:
            self.__agents = agents

        self.__slot_groups = []

        self.__messages = []

        self.__message_loss = message_loss

    def add_agent(self, agent):
        """adds agent to network

        Parameters
        ----------
        agent:
            agent to be added
        """
        self.__agents.append(agent)

    def remove_agent(self, agent):
        """removes agent from list
        if agents is not in list does nothing

        Parameters
        ----------
        agent:
            agent to be removed
        """
        if agent in self.__agents:
            self.__agents.remove(agent)

    def add_slot_group(self, slot_group):
        """adds new slot group
        Parameter
        ---------
            slot_group: SlotGroup
                slot group to be added
        """
        self.__slot_groups.append(slot_group)

    def step(self):
        """Performs one communication round"""

        # select agents randomly, whose messages should be lost (both sending and receiving) in this round
        messages_lost = np.random.choice([1, 0], len(self.__agents), p=[self.__message_loss, 1 - self.__message_loss])

        # send messages of current round, empty in first round
        for message in self.__messages:
            for agent in self.__agents:
                if bool(messages_lost[agent.ID]):
                    agent.send_message(None)
                else:
                    agent.send_message(message)

        # tell agents that communication round is finished
        for agent in self.__agents:
            # in this function the agents should do short time calculations before the next round like priority
            # calculations based on the current measurement
            agent.round_finished()

        # get messages for the next round
        self.__messages = []
        for slot_group in self.__slot_groups:
            for agent_scheduled in slot_group.schedule:

                if bool(messages_lost[agent_scheduled.ID]):
                    self.__messages.append(None)
                else:
                    self.__messages.append(agent_scheduled.get_message(slot_group.id))


        # get schedules for the next round (in reality this computation is done during the round
        # (collection of priorities) and after the round (scheduling for following round))
        for slot_group in self.__slot_groups:
            slot_group.clear_schedule()
            for agent in self.__agents:
                if slot_group.id in agent.slot_group_id_list:
                    slot_group.add_application_for_schedule(agent)

        for agent in self.__agents:
            # in this function the agents should do long time calculations during the current communication round
            agent.round_started()


class Agent(ABC):
    """this class represents an agent inside the network.
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

    def __init__(self, ID, slot_group_id_list):
        """

        Parameters
        ----------
            ID:
                identification number of agent
            slot_group_id_list:
                the list of the ID of the slot group the agent belongs to
        """
        self.__ID = ID
        self.__slot_group_id_list = slot_group_id_list

    def get_prio(self, slot_group_id):
        """returns the priority for the schedulder

        Returns
        -------
            prio:
                priority
            slot_group_id:
                the slot group id the prio should be calculated to
        """
        return 0

    def get_message(self, slot_group_id):
        """returns the current message that should be send

        Parameters
        ----------
            slot_group_id:
                the slot group id the message should belong to

        Returns
        -------
            message: Message
                Message
        """
        return Message(self.ID, slot_group_id, 0)

    def send_message(self, message):
        """send message to agent.

        Parameters
        ----------
            message: Message
                message to send.
        """
        pass

    def round_finished(self):
        """this function has to be called at the end of the round to tell the agent that the communication round is
        finished"""
        pass

    def round_started(self):
        """this function is called automatically by the network when the current communication round starts. In this
            Function longer time calculations should be done."""
        pass

    @property
    def ID(self):
        """

        Returns
        -------
        ID: int
            ID of agent
        """
        return self.__ID

    @property
    def slot_group_id_list(self):
        """

        Returns
        -------
        slot_group_id_list: Array
            array containing IDs of the slot groups the agent belongs to.
        """
        return self.__slot_group_id_list


@dataclass
class Message:
    """this class represents a message that should be send across the network

    Paramters
    ---------
        ID: int
            identification number of the agent who has send the message

        slot_group_id:
            the slot group id the message should belong to

        content:
            content of message that should be send
    """
    ID: int
    slot_group_id: int
    content: any
