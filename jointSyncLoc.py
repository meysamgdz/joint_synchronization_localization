
import matplotlib.pyplot as plt
from numpy.linalg import *
import jointSyncLoc as slf
import os
from SyncNet import SyncNet
from helper import *


class jointSyncLoc:
    def __init__(self, v_const, T,  AP_step):
        self.v_const = v_const
        self.T = T
        self.AP_step = AP_step
    def brf_loc_xAP(self, mu_t=1, std=8, num_iter=1, num_AP=1, nlos_id=0):
        """
        This function returns the location of the object using the linearized Bayesian Recursive Filtering (L-BRF)
        and the location of the Access Points (APs).

        Args:
        mu_t (float): Mean value of the time delay of time-stamping.
        std (float): Standard deviation of the time delay.
        num_iter (int): Number of iterations.
        num_AP (int): Number of APs.
        nlos_id (int): Indicator of the line of sight (LOS) or non-line of sight (NLOS) between the object and the APs.

        Returns:
        x_loc (list): The x-coordinate of the location of the object.
        x_off (list): The offset of the location of the object.
        """
        # initilization of the variables
        x_loc = []
        x_off = []

        # determining the location of the ANs
        sat_loc, ind = self.get_AP_loc()
        count = 0.0

        for u in np.arange(num_iter):
            # generating the MU trajectory
            mob_ag = self.get_traj()
            traj_len = len(mob_ag[:, 0])

            # true clock parameteres
            theta_j, gamma_j, num_ex = np.random.normal(0, 1000), 1 + np.random.normal(0, 1e-3), 1

            # BRF matrix initialization
            UE_pos_init = mob_ag[0, :] + np.random.normal(0, 1, 2)
            prev_state = np.array([[1], [0], [UE_pos_init[0]], [UE_pos_init[1]]])
            xy_noise = 10 * self.T

            # PV model
            P_n = np.diag([1e-3, 1e+7, 1e+6, 1e+6])
            Q = np.array([[1e-7, 0, 0, 0], [0, 1e1, 0, 0],
                          [0, 0, xy_noise ** 2, 0.0], [0, 0, 0.0, xy_noise ** 2]])
            # NLoS identifier is based on K-Rice factor and around 80%
            nlos_id_vec = np.random.binomial(1, 0.801, size=(traj_len, num_AP))
            for i in np.arange(0, traj_len):
                print(f'L-BRF: Step {i} out of {traj_len} steps in the trajectory')
                los_true = np.zeros(num_AP)
                los_det = los_true
                if nlos_id == 1:
                    while np.all(los_det == 0):
                        los_true = np.random.binomial(1, 0.8, size=(1, num_AP))[0]
                        los_det = ~(los_true ^ nlos_id_vec[i, :]) + 2  # XNOR
                else:
                    los_true = np.ones(num_AP)
                    los_det = los_true

                v_c = 1 / 0.3
                UE_pos = mob_ag[i]
                mu_v = mu_t + (los_true == 0) * np.abs(np.random.normal(2, 2, 1))
                ts_dict = {}
                dist = np.sum((UE_pos - sat_loc) ** 2, axis=1) ** 0.5
                dist_sort = np.sort(dist)

                network = slf.SyncNet()
                AP_pos = np.empty((0, 2))
                logic1 = []
                # checking whether an AP is on the x or y-axis. Plus, collecting time-stamps exchanged btw the UE and
                # the APs
                for k in np.arange(num_AP):
                    temp = int(np.where(dist_sort[k] == dist)[0])
                    AP_pos = np.append(AP_pos, np.expand_dims(sat_loc[temp], axis=0), axis=0)

                    # logical vector for determining the AoA estimation variance
                    logic1 += [1 * np.any(ind == temp)]

                    # retrieving the offset and skew of Access Points conducting joint sync&loc
                    relative_off, _, __ = network.synchronize()
                    c_ts_AP_ij = self.timestamp_exchange(relative_off, theta_j, 1.0, gamma_j, AP_pos[k], UE_pos,
                                                         mu_v[k], std, num_ex)
                    ts_dict.update({str(k): c_ts_AP_ij})
                logic1 = np.array(logic1)

                # Prediction step
                F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
                curr_state = multi_dot([F, prev_state])  # + np.random.normal(0, np.diag(Q), size=(1, 4)).T
                P_n = multi_dot([F, P_n, F]) * np.identity(4) + Q

                # Observation step
                mob_loc_pred = curr_state.T[0, 2:4]

                # AoA calculations & Taylor expansion for AoA equations
                der_AoA, AoA_0 = der_arctan(mob_loc_pred, AP_pos)

                delta_xy = mob_loc_pred - AP_pos
                der_dist = (v_c * delta_xy.T / np.sum(delta_xy ** 2, axis=1) ** 0.5).T
                dist = v_c * np.sum(delta_xy ** 2, axis=1) ** 0.5

                diff = (UE_pos - AP_pos).T
                AoA_true = np.arctan2(diff[1], diff[0])
                logic2 = (np.abs(AoA_true) > np.pi / 2) * 1
                # ang_std = slf.CRB_AoA(UE_pos, AP_pos, AoA_true, logic1)
                # AoA_meas = AoA_true + (np.random.normal(0, ang_std) +
                #                        (los_true == 0)*np.abs(np.random.normal(1/18*np.pi, 1/90*np.pi)))
                AoA_temp = (logic1 == 0) * 1 * AoA_true + (logic1 == 1) * logic2 * 1 * (np.abs(AoA_true))
                ang_std = aoa_std(AoA_temp / np.pi * 180)/ 180 * np.pi
                AoA_meas = AoA_true + np.random.normal(0, ang_std) + (los_true == 0) * np.abs(
                    np.random.normal(1 / 18 * np.pi, 1 / 90 * np.pi))

                ind_los = np.where(los_det == 1)[0]
                num_AP_active = AP_pos[los_det == 1].shape[0]

                # observation matrices calculations
                for k in np.arange(num_AP_active):
                    c_ts_AP_ij = ts_dict[str(ind_los[k])]
                    AoA_diff = AoA_meas[ind_los[k]] - AoA_0[ind_los[k]]
                    B = np.array([[c_ts_AP_ij[3, 0] - c_ts_AP_ij[1, 0], 0.0, 0.0, 0.0],
                                  [c_ts_AP_ij[3, 0] + c_ts_AP_ij[4, 0], -2.0, 0.0, 0.0],
                                  [c_ts_AP_ij[4, 0], -1.0, der_dist[ind_los[k], 0], der_dist[ind_los[k], 1]],
                                  [0.0, 0.0, der_AoA[ind_los[k], 0], der_AoA[ind_los[k], 1]]])
                    mu = np.array([[c_ts_AP_ij[2, 0] - c_ts_AP_ij[0, 0]],
                                   [c_ts_AP_ij[2, 0] + c_ts_AP_ij[5, 0]],
                                   [c_ts_AP_ij[5, 0] + np.sum(der_dist[ind_los[k]] * mob_loc_pred) - dist[ind_los[k]]],
                                   [-np.sign(AoA_diff) * 2 * np.pi * (np.abs(AoA_diff) > np.pi)
                                    + AoA_diff + np.sum(der_AoA[ind_los[k]] * mob_loc_pred)]])

                    R0 = np.diag(np.array([2 * std ** 2, 2 * std ** 2, std ** 2, (1.2 / 180 * np.pi) ** 2]))
                    B_t = np.dot(B.T, B)
                    z = multi_dot([inv(B_t), B.T, mu])
                    R = multi_dot([inv(B_t), B.T, R0, B, inv(B_t).T])
                    R = np.diag(R) * np.identity(4)

                    # Update
                    curr_state = np.dot(inv(R + P_n), np.dot(R, curr_state) + np.dot(P_n, z))
                    P_n = inv(inv(R) + inv(P_n))

                # plt.plot(curr_state[2], curr_state[3], 'r *', linewidth=2)
                error = np.sum((mob_ag[i, :] - curr_state.T[0, 2:4]) ** 2) ** 0.5

                if i > 20:
                    if error < 20:
                        # x_off = x_off + [np.abs(theta_j-curr_state[1]/curr_state[0]-0.5*(theta_i+theta_l))]
                        x_off = x_off + [np.abs(theta_j - curr_state[1] / curr_state[0])]
                        theta_j = theta_j - (curr_state[1] / curr_state[0]) + np.random.normal(0, 0.5)
                        gamma_j = gamma_j - 1 / curr_state[0] + 1.0 + np.random.normal(0, 1e-7)
                        prev_state = curr_state
                        prev_state[0] = 1
                        prev_state[1] = 0
                    else:
                        count += 1
                        curr_state.T[0, 2:4] = mob_ag[0, :] + np.random.normal(0, 2, 2)
                        prev_state = curr_state
                        prev_state[0] = 1
                        prev_state[1] = 0
                    x_loc = x_loc + [error]
        print('An2', count)
        return np.array(x_loc), np.array(x_off)
    def PF_loc_xAP(self, mu_t=1, std=8, num_iter=1, num_particles=500, num_AP=1, nlos_id=0):
        """
            This function uses the hybrid parametric and particle filter algorithm to estimate the clock and location
            parameters f the user equipment (UE) while taking into account the location of the access points (APs).

            Args:
            - mu_t (float): The mean of the delay in time-stamp exchange between the UE and the APs. Default is 1.
            - std (float): The standard deviation of time-stamp exchange between the UE and the APs. Default is 8.
            - num_iter (int): The number of iterations for the particle filter. Default is 1.
            - num_particles (int): The number of particles used in the particle filter. Default is 500.
            - num_AP (int): The number of access points considered in the simulation. Default is 1.
            - nlos_id (int): A binary variable that indicates whether to use the NLoS-identifier system or not.
                             0 for not using, 1 for using. Default is 0.

            Returns:
            - np.array(x_loc): The location estimation error for each trajectory point
            - np.array(x_off): The offset estimation error for each trajectory point
            - np.array(N_eff): Number of effective particles monitored throughout the simulation
            """
        # initilization of the variables
        mu_t = np.array([mu_t])
        x_off = []
        x_loc = []

        # determining the location of the APs
        x_range = (0, 500)
        y_range = (0, 500)
        sat_loc, ind = self.get_AP_loc()
        count = 0.0
        for u in np.arange(num_iter):
            # generating the MU trajectory
            mob_ag = self.get_traj()
            traj_len = len(mob_ag[:, 0])

            # true clock parameters obtained using the hyb sync algorithm
            theta_j, gamma_j, num_ex = np.random.normal(0, 1000), 1 + np.random.normal(0, 1e-3), 1

            # Particle initialization
            particles = create_uniform_particles(x_range, y_range, num_particles)
            weights = np.ones(num_particles)
            P_n = np.array([[1e-3, 0.], [0., 1e+7]])
            Q_n = np.array([[1e-7, 0.], [0., 1e1]])
            prev_state = np.array([[1], [0]])
            xy_noise = 10 * self.T
            sig_p_noise = 5 * num_particles ** (-1 / 5)
            P_n_loc = sig_p_noise ** 2 * np.tile(np.array([[1, 0.], [0., 1]]), (num_particles, 1, 1))

            N_eff = []
            # NLoS identifier is based on DNN and its accuracy is 95%
            nlos_id_vec = np.random.binomial(1, 0.951, size=(traj_len, num_AP))
            for i in np.arange(1, traj_len):
                print(f'PF: Step {i} out of {traj_len} steps in the trajectory')
                # determine whether we use NLoS-identifier system or not & the number of APs involved in joint sync&loc
                los_det = np.zeros(num_AP)
                if nlos_id == 1:
                    while np.all(los_det == 0):
                        los_true = np.random.binomial(1, 0.8, size=(1, num_AP))[0]
                        los_det = ~(los_true ^ nlos_id_vec[i, :]) + 2  # XNOR
                else:
                    los_true = np.ones(num_AP)
                    los_det = los_true

                # Retrieving the indices of the closest APs and their respective clock parameters
                network = slf.SyncNet()
                # particle_noise = np.random.normal(0, sig_p_noise, size=(num_particles, 2))
                UE_pos = mob_ag[i, :]
                mu_v = mu_t + (los_true == 0) * np.abs(np.random.normal(2, 2, 1))
                ts_dict = {}
                dist = np.sum((UE_pos - sat_loc) ** 2, axis=1) ** 0.5
                dist_sort = np.sort(dist)
                AP_pos = np.empty((0, 2))
                logic1 = []
                for k in range(num_AP):
                    temp = int(np.where(dist_sort[k] == dist)[0])
                    AP_pos = np.append(AP_pos, np.expand_dims(sat_loc[temp], axis=0), axis=0)

                    # logical vector for determining the AoA estimation variance
                    logic1 = logic1 + [1 * np.any(ind == temp)]

                    # retrieving the offset and skew of Access Points conducting joint sync&loc
                    relative_off, _, __ = network.synchronize()
                    c_ts_AP_ij = self.timestamp_exchange(relative_off, theta_j, 1.0, gamma_j, AP_pos[k], UE_pos,
                                                         mu_v[k], std, num_ex, asym=True)
                    ts_dict.update({str(k + 1): c_ts_AP_ij})
                logic1 = np.array(logic1)
                #A. PREDICTION step
                curr_state = prev_state
                P_n = P_n + Q_n
                noise_std = np.array([xy_noise, xy_noise])
                particles += np.random.normal(0, noise_std, size=(num_particles, 2))
                # P_n_loc += Q_n_loc

                # B UPDATE step based on measurements
                # B.1 AoA calculated by each AP
                v_c = 1 / 0.3
                diff = (UE_pos - AP_pos).T
                AoA_true = np.arctan2(diff[1], diff[0])
                AoA_true = AoA_true * 1 * (AoA_true > 0) + (AoA_true + 2 * np.pi) * 1 * (AoA_true < 0)
                # ang_std = slf.CRB_AoA(UE_pos, AP_pos, AoA_true, logic1) + (los_true == 0)*np.abs(np.random.normal(10, 2))
                # AoA_meas = AoA_true + np.random.normal(0, ang_std)
                logic2 = (np.abs(AoA_true) > np.pi / 2) * 1
                AoA_temp = (logic1 == 0) * 1 * AoA_true + (logic1 == 1) * logic2 * 1 * (np.abs(AoA_true))
                ang_std = aoa_std(AoA_temp / np.pi * 180) / 180 * np.pi
                AoA_meas = AoA_true + np.random.normal(0, ang_std) + (los_true == 0) * np.abs(
                    np.random.normal(1 / 18 * np.pi, 1 / 90 * np.pi))

                # AP_pos = AP_pos[los_det.astype(int)]
                ind_los = np.where(los_det == 1)[0]
                num_AP_active = AP_pos[los_det == 1].shape[0]
                weights.fill(1.)
                for k in np.arange(num_AP_active):
                    # Building up measurement matrices based on time-stamps
                    c_ts_AP_ij = ts_dict[str(ind_los[k] + 1)]
                    B = np.array([[c_ts_AP_ij[3, 0] - c_ts_AP_ij[1, 0], 0.0],
                                  [c_ts_AP_ij[3, 0] + c_ts_AP_ij[4, 0], -2.0]])
                    mu = np.array([[c_ts_AP_ij[2, 0] - c_ts_AP_ij[0, 0]],
                                   [c_ts_AP_ij[2, 0] + c_ts_AP_ij[5, 0]]])
                    B_p = np.array([c_ts_AP_ij[4, 0], -1.0])
                    mu_p = np.array([c_ts_AP_ij[5, 0]])
                    R0 = np.diag(np.array([2 * std ** 2, 2 * std ** 2]))

                    # Clock parameter update step
                    B_t = np.dot(B.T, B)
                    z_t = multi_dot([inv(B_t), B.T, mu])
                    R = multi_dot([inv(B_t), B.T, R0, B, inv(B_t).T])
                    R = np.diag(R) * np.identity(2)

                    curr_state = np.dot(inv(R + P_n), np.dot(R, curr_state) + np.dot(P_n, z_t))
                    P_n = inv(inv(R) + inv(P_n))

                    # Weight update
                    diff = (particles - AP_pos[k]).T
                    AoA_particle = np.arctan2(diff[1], diff[0])
                    AoA_particle[AoA_particle < 0] = AoA_particle[AoA_particle < 0] + 2 * np.pi
                    AoA_diff = np.abs(AoA_particle - AoA_meas[ind_los[k]])
                    weights *= (1 / AoA_diff) / np.sum(1 / AoA_diff)
                    weights += 1.e-100
                    weights /= np.sum(weights)
                    dist_error = np.abs(
                        np.dot(B_p, z_t) - mu_p + v_c * np.sum((particles - AP_pos[ind_los[k]]) ** 2, axis=1) ** 0.5)
                    weights *= np.exp(-dist_error) / np.sum(np.exp(-dist_error))
                    weights += 1.e-100  # avoid round-off to zero
                    weights /= np.sum(weights)  # normalize

                    # Location update -> particle gaussian mixture
                    delta_xy = (particles - AP_pos[ind_los[k]]).T
                    temp = delta_xy / np.sum(delta_xy ** 2, axis=0)
                    temp[1, :] = -temp[1, :]
                    der_AoA = temp[::-1, :].T
                    AoA_0 = np.arctan2(delta_xy[1], delta_xy[0])
                    AoA_diff = AoA_meas[ind_los[k]] - AoA_0

                    der_dist = (v_c * delta_xy / np.sum(delta_xy ** 2, axis=0) ** 0.5).T
                    dist = v_c * np.sum(delta_xy ** 2, axis=0) ** 0.5

                    B = np.array([[der_AoA[:, 0], der_AoA[:, 1]],
                                  [der_dist[:, 0], der_dist[:, 1]]])
                    mu = np.array([[-np.sign(AoA_diff) * 2 * np.pi * (np.abs(AoA_diff) > np.pi)
                                    + AoA_diff + np.sum(der_AoA * particles, axis=1)],
                                   [np.dot(B_p, z_t) - c_ts_AP_ij[5, 0] +
                                    np.sum(der_dist * particles, axis=1) - dist]])
                    R0 = np.tile(np.diag(np.array([(1.2 / 180 * np.pi) ** 2, std ** 2])), (num_particles, 1, 1))
                    B = B.transpose(2, 0, 1)
                    mu = mu.transpose(2, 0, 1)
                    B_t = np.matmul(B.transpose(0, 2, 1), B)
                    z = np.matmul(np.matmul(inv(B_t), B.transpose(0, 2, 1)), mu)
                    R = np.matmul(np.matmul(inv(B_t), B.transpose(0, 2, 1)), R0)
                    R = np.matmul(np.matmul(R, B), inv(B_t).transpose(0, 2, 1))
                    R = R * np.tile(np.identity(R.shape[1]), (num_particles, 1, 1))

                    particles = np.matmul(inv(R + P_n_loc),
                                          np.matmul(R, np.expand_dims(particles, axis=2)) + np.matmul(P_n_loc, z))[:, :,
                                0]
                    P_n_loc = inv(inv(R) + inv(P_n_loc))
                x_off = x_off + [np.abs(theta_j - (curr_state[1] / curr_state[0]))]
                theta_j = theta_j - (curr_state[1] / curr_state[0]) + np.random.normal(0, 0.5)
                gamma_j = gamma_j - 1 / curr_state[0] + 1.0 + np.random.normal(0, 1e-7)
                prev_state = np.array([[1], [0]])

                N_eff += [1 / np.sum(weights ** 2)]
                particle_noise = 0

                # B.4 Resampling the particles in the positions with higher prob
                if N_eff[-1] < num_particles * 2 / 3:
                    particles = resample(particles, particle_noise, weights, num_particles)

                error = np.sum((mob_ag[i, :] - np.mean(particles.T, axis=1)) ** 2) ** 0.5
                if i > 50:
                    x_loc = x_loc + [error]

        return np.array(x_loc), np.array(x_off), np.array(N_eff)  # (x_off/(num_iter*traj_len-count))**0.5
    def get_AP_loc(self, distance=200, x_init_traj=10, x_init_AN=30, y_init=200):
        """
        This function returns the location of Access Point(AP) and indices of the APs on the y axis (This is useful to
        know when calculating the AoA estimation error because the antenna aperture is 90 rototed.).

        Args:
        distance (int): The distance between the APs.
        x_init_traj (int): The initial x-coordinate of the trajectory.
        x_init_AN (int): The initial x-coordinate of the AN.
        y_init (int): The initial y-coordinate.

        Returns:
        sat_loc (numpy array): The location of the APs.
        ind (numpy array): Indices of the APs on the y axis.
        """
        dist_to_mid_str = 5
        grid = np.arange(x_init_AN, 2 * distance, self.AP_step)
        grid_p = grid + self.AP_step / 2
        sat_loc_temp_1 = np.array([grid_p, (y_init + dist_to_mid_str) * np.ones(len(grid))]).T
        sat_loc_temp_2 = np.array([grid, (y_init - dist_to_mid_str) * np.ones(len(grid))]).T
        sat_loc = np.append(sat_loc_temp_1, sat_loc_temp_2, axis=0)
        sat_loc_temp = np.array([grid_p, (y_init + dist_to_mid_str + distance) * np.ones(len(grid))]).T
        sat_loc = np.append(sat_loc, sat_loc_temp, axis=0)
        sat_loc_temp = np.array([grid, (y_init - dist_to_mid_str + distance) * np.ones(len(grid))]).T
        sat_loc = np.append(sat_loc, sat_loc_temp, axis=0)
        sat_loc_temp = np.array([grid_p, (y_init + dist_to_mid_str - distance) * np.ones(len(grid))]).T
        sat_loc = np.append(sat_loc, sat_loc_temp, axis=0)
        sat_loc_temp = np.array([grid, (y_init - dist_to_mid_str - distance) * np.ones(len(grid))]).T
        sat_loc = np.append(sat_loc, sat_loc_temp, axis=0)
        ind_temp = np.where(grid > distance + x_init_traj)
        ind_temp = ind_temp[0][0]
        sat_loc_temp = np.array([[grid[ind_temp - 1], y_init + dist_to_mid_str + (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp - 1],
                                  y_init + dist_to_mid_str + 2 * (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp], y_init + dist_to_mid_str + (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp], y_init + dist_to_mid_str + 2 * (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp - 1], y_init - dist_to_mid_str - (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp - 1],
                                  y_init - dist_to_mid_str - 2 * (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp], y_init - dist_to_mid_str - (distance - 2 * dist_to_mid_str) / 3],
                                 [grid[ind_temp], y_init - dist_to_mid_str - 2 * (distance - 2 * dist_to_mid_str) / 3]])
        sat_loc = np.append(sat_loc, sat_loc_temp, axis=0)
        ind = np.append(np.expand_dims(np.where(grid[ind_temp] == sat_loc.T[0, :])[0], axis=0),
                        np.expand_dims(np.where(grid[ind_temp - 1] == sat_loc.T[0, :])[0], axis=0), axis=1)[0]
        sat_loc = sat_loc + np.random.normal(0, 0.1, np.shape(sat_loc))
        return sat_loc, ind
    def get_traj(self, distance=200, x_init=10, y_init=200):
        """
        This function returns the trajectory of the object in 2D space.

        Args:
        distance (int): The distance between the APs.
        x_init (int): The initial x-coordinate of the trajectory.
        y_init (int): The initial y-coordinate.

        Returns:
        x (numpy array): The x-coordinates of the trajectory.
        y (numpy array): The y-coordinates of the trajectory.
        """

        # random acceleration
        a_xy = 1.5 * np.random.random_sample(1)[0] + 1

        # time interval of the 1st and 3rd sections of the trajectory
        t_v = self.v_const / a_xy
        # time interval of the 2nd section of the trajectory
        t0 = (distance - a_xy * t_v ** 2) / self.v_const
        t1 = np.arange(0, t0, self.T)

        # time steps for the 1st and 3rd sections of the trajectory
        acc_t_1 = np.arange(0, t_v, self.T)
        acc_t_2 = acc_t_1[::-1]

        # generating the 1st part
        x1_1 = +0.5 * a_xy * acc_t_1 ** 2 + x_init
        x1_2 = +self.v_const * t1 + x1_1[-1]
        x1_3 = (0.5 * a_xy * acc_t_2 ** 2)[::-1] + x1_2[-1]
        x1 = np.concatenate([x1_1, x1_2, x1_3])
        y1 = y_init * np.ones(len(t1) + 2 * len(acc_t_1))

        # taking the turn randomly
        intersection = np.random.randint(0, 2, 1)
        tmp = np.array([-1, 1])
        intersection = tmp[intersection]
        a_xy = 1.5 * np.random.random_sample(1)[0] + 1
        t_v = self.v_const / a_xy
        t0 = (distance - a_xy * t_v ** 2) / self.v_const
        t1 = np.arange(0, t0, self.T)
        acc_t_1 = np.arange(0, t_v, self.T)
        acc_t_2 = acc_t_1[::-1]

        # generating the 2nd part
        x2 = x1[-1] * np.ones(len(t1) + 2 * len(acc_t_1))
        y2_1 = intersection * 0.5 * a_xy * acc_t_1 ** 2 + y1[-1]
        y2_2 = intersection * self.v_const * t1 + y2_1[-1]
        y2_3 = intersection * (0.5 * a_xy * acc_t_2 ** 2)[::-1] + y2_2[-1]
        y2 = np.concatenate([y2_1, y2_2, y2_3])

        intersection = np.random.randint(0, 2, 1)
        tmp = np.array([-1, 1])
        intersection = tmp[intersection]

        a_xy = 1.5 * np.random.random_sample(1)[0] + 1
        t_v = self.v_const / a_xy
        t0 = (distance - a_xy * t_v ** 2) / self.v_const
        t1 = np.arange(0, t0, self.T)
        acc_t_1 = np.arange(0, t_v, self.T)
        acc_t_2 = acc_t_1[::-1]

        # generating the 3rd part
        x3_1 = intersection * 0.5 * a_xy * acc_t_1 + x2[-1]
        x3_2 = intersection * self.v_const * t1 + x3_1[-1]
        x3_3 = 0.5 * (a_xy * acc_t_2 ** 2)[::-1] + x3_2[-1]
        x3_3 = intersection * self.v_const * acc_t_1 + x3_2[-1]
        x3 = np.concatenate([x3_1, x3_2, x3_3])
        y3 = y2[-1] * np.ones(len(t1) + 2 * len(acc_t_1))

        # merging the 3 parts
        x_t = np.concatenate([x1, x2, x3])
        y_t = np.concatenate([y1, y2, y3])

        mob_ag = np.array([x_t, y_t]).T
        return mob_ag

    @staticmethod
    def timestamp_exchange(the_i, the_j, gamma_i, gamma_j, AP_pos, UE_pos, const_delay, std, num_ex, asym=True):
        """
        This is a Python function that simulates timestamps for a communication event between two devices, labeled as i and
        j. It takes in several inputs: the initial time (the_i, the_j), the rate of time drift for each device
        (gamma_i, gamma_j), the number of iterations (num_ex) to run the simulation, and the standard deviation (std) of
        the egress and ingress time (T_n and R_n) which are modeled as random normal variables. It returns a 4xiter array
        of timestamps, where each column represents a single iteration of the simulation and the rows correspond to
        different events (egress from i, ingress to j, egress from j, ingress to i). The function also includes a
        fixed waiting time, distance between the devices, and a fixed time (T) between iterations.

        Args:
        - theta_i (float): Initial timestamp for device i
        - theta_j (float): Initial timestamp for device j
        - gamma_i (float): Rate of time drift for device i
        - gamma_j (float): Rate of time drift for device j
        - num_ex (int): Number of iterations to run the simulation
        - std (float): Standard deviation of the egress and ingress time

        Returns:
        - C_stamp (numpy array): The exchanged time-stamps
        """
        sigma_eg = std
        sigma_ing = std
        light_v = 0.3  # m/ns
        d_ij = np.sum((UE_pos - AP_pos) ** 2) ** 0.5 / light_v
        T = (1 / 128) * 1e+9
        waiting_time0 = 0.999 * T
        waiting_time = 2 * d_ij
        iter = num_ex
        if asym:
            C_stamp = np.zeros((6, iter))
            for i in range(iter):
                t = i * T
                # egress and ingress time (are generally random)
                T_n0 = np.abs(np.random.normal(const_delay, sigma_eg))
                T_n = np.abs(np.random.normal(const_delay, sigma_eg))
                R_n = np.abs(np.random.normal(const_delay, sigma_ing))
                C_stamp[0, i] = gamma_i * (t + waiting_time) + the_i
                C_stamp[1, i] = gamma_j * (t + waiting_time + d_ij + T_n0) + the_j
                C_stamp[2, i] = gamma_i * (t + waiting_time + waiting_time0) + the_i
                C_stamp[3, i] = gamma_j * (t + waiting_time + waiting_time0 + d_ij + T_n) + the_j
                C_stamp[4, i] = gamma_j * (t + waiting_time + waiting_time0 + d_ij + T_n + waiting_time) + the_j
                C_stamp[5, i] = gamma_i * (
                        t + waiting_time + waiting_time0 + d_ij + T_n + waiting_time + d_ij + R_n) + the_i
        else:
            C_stamp = np.zeros((4, iter))
            for i in np.arange(iter):
                t = i * T
                # egress and ingress time (are generally random)
                T_n = np.random.normal(const_delay, sigma_eg)
                R_n = np.random.normal(const_delay, sigma_ing)
                C_stamp[0, i] = gamma_i * (t + waiting_time) + the_i
                C_stamp[1, i] = gamma_j * (t + waiting_time + d_ij + T_n) + the_j
                C_stamp[2, i] = gamma_j * (t + waiting_time + d_ij + T_n + waiting_time) + the_j
                C_stamp[3, i] = gamma_i * (t + waiting_time + d_ij + T_n + waiting_time + d_ij + R_n) + the_i
        return C_stamp

    def retrieve_ap_off_skew(bp_flag, std):
        if bp_flag[0] == 1:
            rand_num = np.random.randint(0, 1000)
            temp1 = np.load('./results_save/true_off_bp_' + bp_flag[1] + 'th.npy')
            true_off = temp1[2 * (int(std) - 1):2 * int(std), rand_num]
            temp2 = np.load('./results_save/true_skew_bp_' + bp_flag[1] + 'th.npy')
            true_skew = 1.0 + temp2[2 * (int(std) - 1):2 * int(std), rand_num]
            theta_1, theta_2, gamma_1, gamma_2 = true_off[0], true_off[1], true_skew[0], true_skew[1]
            rand_num = np.random.randint(0, 1000)
            true_off = temp1[2 * (int(std) - 1):2 * int(std), rand_num]
            true_skew = 1.0 + temp2[2 * (int(std) - 1):2 * int(std), rand_num]
            theta_3, theta_4, gamma_3, gamma_4 = true_off[0], true_off[1], true_skew[0], true_skew[1]
            cl_param = {'theta_1': theta_1, 'theta_2': theta_2, 'theta_3': theta_3, 'theta_4': theta_4,
                        'gamma_1': gamma_1, 'gamma_2': gamma_2, 'gamma_3': gamma_3, 'gamma_4': gamma_4}
        else:
            rand_num = np.random.randint(0, 1000)
            temp1 = np.load('./results_save/true_off_hyb_' + bp_flag[1] + 'th.npy')
            true_off = temp1[2 * (int(std) - 1):2 * int(std), rand_num]
            temp2 = np.load('./results_save/true_skew_hyb_' + bp_flag[1] + 'th.npy')
            true_skew = 1.0 + temp2[2 * (int(std) - 1):2 * int(std), rand_num]
            theta_1, theta_2, gamma_1, gamma_2 = true_off[0], true_off[1], true_skew[0], true_skew[1]
            rand_num = np.random.randint(0, 1000)
            true_off = temp1[2 * (int(std) - 1):2 * int(std), rand_num]
            true_skew = 1.0 + temp2[2 * (int(std) - 1):2 * int(std), rand_num]
            theta_3, theta_4, gamma_3, gamma_4 = true_off[0], true_off[1], true_skew[0], true_skew[1]
            rand_num = np.random.randint(0, 1000)
            true_off = temp1[2 * (int(std) - 1):2 * int(std), rand_num]
            true_skew = 1.0 + temp2[2 * (int(std) - 1):2 * int(std), rand_num]
            theta_5, theta_6, gamma_5, gamma_6 = true_off[0], true_off[1], true_skew[0], true_skew[1]
            cl_param = {'theta_1': theta_1, 'theta_2': theta_2, 'theta_3': theta_3, 'theta_4': theta_4,
                        'theta_5': theta_5, 'theta_6': theta_6,
                        'gamma_1': gamma_1, 'gamma_2': gamma_2, 'gamma_3': gamma_3, 'gamma_4': gamma_4,
                        'gamma_5': gamma_5, 'gamma_6': gamma_6}
        return cl_param
