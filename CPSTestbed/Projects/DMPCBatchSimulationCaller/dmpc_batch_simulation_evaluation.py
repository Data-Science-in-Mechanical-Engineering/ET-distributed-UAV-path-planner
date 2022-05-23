import pickle as p
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import CPSTestbed.useful_scripts.plotter

sys.path.append("../../../gym-pybullet-drones")
sys.path.append("../../../")


def save_dict(file, dict):
    with open(file) as f:
        for key in dict.keys():
            f.write(str(key) + " " + str(dict[key]) + "\n")


def save(file, dist_to_target, timestamps):
    max_t = timestamps[list(dist_to_target.keys())[0]][-1]
    for key in dist_to_target.keys():
        if max_t < timestamps[key][-1]:
            max_t = timestamps[key][-1]
    sampling = 1
    for key in dist_to_target.keys():
        y = [np.mean(dist_to_target[key][:, i]) for i in range(len(timestamps[key]))]
        x = timestamps[key]

        with open(os.path.join(file + "_" + str(key) + ".csv"), "w") as f:
            for i in range(len(x)):
                if i%sampling == 0 :#and x[i] <= 50:
                    f.write(str(x[i]) + " " + str(y[i]) + "\n")
            f.write(str(x[-1]) + " " + str(y[-1]) + "\n")
            f.write(str(max_t) + " " + str(y[-1]) + "\n")


def save_traj(file, traj, timestamps):
    i = len(timestamps[0]) - 1
    while i >= 0:
        if timestamps[0, i] != 0:
            break
        i -= 1
    ending = i + 1
    i = len
    sampling = 1
    with open(os.path.join(file), "w") as f:
        for i in range(ending):
            if i%sampling == 0:
                f.write(str(traj[0, i]) + " " + str(traj[1, i]) + " " + str(traj[2, i]) + "\n")


def time_to_target(simulation_results):
    dist_to_target = {}
    timestamps = {}
    asdf = 0
    for result in simulation_results:
        print(asdf)
        asdf += 1
        # take any timestamp since they are identical (same simulation_frequency)
        i = len(result.timestamps[0, :]) - 1
        while i >= 0:
            if result.timestamps[0, i] != 0:
                break
            i -= 1
        ending = i+1
        result.timestamps = result.timestamps[:, 0:ending]
        if result.num_drones not in timestamps.keys():
            timestamps[result.num_drones] = result.timestamps[0, 0:ending]
            #for t in timestamps[result.num_drones]:
            #    print(t)
        else:
            if len(timestamps[result.num_drones]) < len(result.timestamps[0, :]):
                dl = len(result.timestamps[0, :]) - len(timestamps[result.num_drones])
                timestamps[result.num_drones] = result.timestamps[0, :]
                #for t in timestamps[result.num_drones]:
                #    print(t)
                dist_to_target[result.num_drones] = np.concatenate((dist_to_target[result.num_drones],
                                                                   np.tile(dist_to_target[result.num_drones][:, -1], (dl, 1)).T), axis=1)
        # calc dist to target over time of each drone and store it
        for i in range(result.num_drones):
            pos = result.states[i, 0:3, :]
            target = result.targets[i]
            #crashed = result.crashed[i]

            # don't use this drone if it crashed
            #if hasattr(crashed, "__len__"):
            #    if any(crashed):
            #        continue

            dist = [np.linalg.norm(pos[:, j] - np.transpose(target)) if j < ending else np.linalg.norm(pos[:, ending-1] - np.transpose(target)) for j in
                    range(max(ending, len(timestamps[result.num_drones])))]

            if result.num_drones in dist_to_target.keys():
                dist_to_target[result.num_drones] = np.vstack((dist_to_target[result.num_drones],
                                                               dist))
            else:
                dist_to_target[result.num_drones] = dist

    return timestamps, dist_to_target


path = ""
plot_states = False

files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.startswith("ARGS")]

if not plot_states:
    successrate_total = {}
    successrate = {}
    number_sims = {}
    number_sims_agents = {}
    number_crashes = {}
    simulation_results = []
    number_crashes_total = {}
    for f in files:
        result = p.load(open(f, "rb" ))
        simulation_results.append(result)
        ARGS = result.ARGS
        logger = result
        if ARGS.num_drones not in successrate:
            successrate_total[ARGS.num_drones] = 0
            successrate[ARGS.num_drones] = 0
            number_sims[ARGS.num_drones] = 0
            number_sims_agents[ARGS.num_drones] = 0
            number_crashes[ARGS.num_drones] = 0
            number_crashes_total[ARGS.num_drones] = 0

        success = True
        for s in logger.successful:
            success = success and s
            if s:
                successrate[ARGS.num_drones] += 1
            number_sims_agents[ARGS.num_drones] += 1
        crash = False
        for c in logger.crashed.values():
            if c:
                crash = True
                number_crashes[ARGS.num_drones] += 1
        if crash:
            number_crashes_total[ARGS.num_drones] += 1

        if success:
            successrate_total[ARGS.num_drones] += 1

        number_sims[ARGS.num_drones] += 1
        #if number_sims[ARGS.num_drones] >= 10:
        #    break

    print(successrate_total)
    print(number_sims)
    print(successrate)
    print(number_sims_agents)
    print(number_crashes)
    temp = list(number_crashes.keys())
    temp.sort()
    print([number_crashes[j] / number_sims_agents[j] for j in temp])
    print([number_crashes_total[j] / number_sims[j] for j in temp])
    timestamps, dist_to_target = time_to_target(simulation_results)
    save(os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/Plot/RoundRobinFinal2", dist_to_target, timestamps)

    plt.rc('axes', prop_cycle=cycler('color', ['r', 'g', 'b', 'y', 'm', 'c']))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Average Distance To Target \n ' + str(len(simulation_results)) + ' Simulations')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance To Target [m]')
    ax.grid()

    for key in dist_to_target.keys():
        y = [np.mean(dist_to_target[key][:, i]) for i in range(len(timestamps[key]))]
        x = timestamps[key]
        # plt.errorbar(x, y, err, linestyle='None', fmt='o', c='b', capsize=5)
        ax.plot(x, y, label=str(key) + ' Drones')


    ax.legend()
    plt.show()

else:
    file = files[0]
    result = p.load(open(file, "rb"))
    states = result.states
    controls = result.controls
    timestamps = result.timestamps
    targets = result.targets

    save_file = os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/Plot/Example"
    plot_type = '3D'
    drone_ids = None
    if drone_ids is None:
        drone_ids, _, _ = states.shape
        drone_ids = range(drone_ids)

    if type(drone_ids) == int:
        drone_ids = [drone_ids]

    for i in drone_ids:
        states[i, 0:2, :] = states[i, 0:2, :] - 2
        save_traj(save_file + str(i) + ".csv", states[i, :, :], result.timestamps)

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
