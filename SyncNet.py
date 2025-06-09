import numpy as np
from helper import *
from numpy.linalg import *

class SyncNet:
    def __init__(self, Nodes: list = ['N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6', 'N_7', 'N_8', 'N_9'],
                 KF_nodes: list = ['N_21', 'N_31', 'N_41', 'N_42', 'N_51', 'N_52', 'N_61', 'N_62', 'N_71', 'N_81', 'N_82',
                           'N_91', 'N_92'],
                 Factors: list = ['F_12', 'F_17', 'F_26', 'F_23', 'F_29', 'F_34', 'F_35', 'F_45', 'F_47', 'F_48', 'F_56',
                          'F_58', 'F_69'], syncType: str = 'hybrid'):
        self.Nodes = Nodes
        self.Factors = Factors
        self.KF_nodes = KF_nodes
        self.syncType = syncType

    def brf(self, the_i: float, the_j: float, gamma_i: float, gamma_j: float, i: int, num_ex: int,
            std: float) -> tuple[np.ndarray, np.ndarray]:
        """
        This function implements the Bayesian Recursive Filtering algorithm to estimate the offset and skew of a wireless sensor node.
        The function takes the initial offset and skew of the node, as well as the offset and skew of the node it is communicating with, as inputs.
        It also takes the iteration number, number of timestamps to generate, and standard deviation of the timestamps.
        The function returns the estimated offset and skew of the node, along with the standard deviation of the estimates.

        Parameters:
        - the_i (float): Initial offset of the node
        - the_j (float): Initial offset of the communicating node
        - gamma_i (float): Initial skew of the node
        - gamma_j (float): Initial skew of the communicating node
        - i (int): The iteration number
        - num_ex (int): Number of timestamps to generate
        - std (float): Standard deviation of the timestamps

        Returns:
        - real_clk_skw : the estimated offset and skew for the node
        - np.sqrt(np.diag(P_n)) : the standard deviation of the estimates
        """
        # implementation of bayesian recursive filtering
        time_st = self.timestamp_exchange(the_i, the_j, gamma_i, gamma_j, num_ex, std, asym=False)
        iter = i + 1 + int(np.log2(num_ex / 10))
        R0 = std ** 2 * np.array([[2, 0.0], [0.0, 2]])
        P_n = np.array([[10 * std ** 2, 0.0], [0.0, 1e-2]])
        clk_skw_k = np.array([[0.0], [1]])
        real_clk_skw = np.array([[0.0], [0.0]])
        for k in np.arange(1, iter):
            T = time_st[0, k] - time_st[0, k - 1]
            A = np.array([[1.0, -T], [0.0, 1]])
            B = np.array([[-2, time_st[1, k] + time_st[2, k]], [0, time_st[1, k] - time_st[1, k - 1]]])
            C = np.array([[0.0, T], [0.0, 0.0]])
            # Prediction
            clk_skw_k = np.dot(A, clk_skw_k) + np.dot(C, clk_skw_k)
            P_n = multi_dot([A, P_n, A.T])
            P_n = np.diag(P_n) * np.identity(2)
            # Measurment
            z = np.array([[time_st[0, k] + time_st[3, k]], [T]])
            z = np.dot(inv(B), z)
            R = multi_dot([inv(B), R0, inv(B).T])
            R = np.diag(R) * np.identity(2)
            # correction
            K = np.dot(P_n, inv(R + P_n))
            clk_skw_k = np.dot(inv(R + P_n), np.dot(R, clk_skw_k) + np.dot(P_n, z))
            P_n = inv(inv(R) + inv(P_n))
        real_clk_skw[0] = clk_skw_k[0] / clk_skw_k[1]
        real_clk_skw[1] = 1 / clk_skw_k[1]
        return real_clk_skw, np.sqrt(np.diag(P_n))
    def synchronize(self, sim_iter: int = 1, low: float = -1e3, high: float = 1e3, GrandMaster: str = '1',
                    L_bp_iter: int = 8, num_ex: int = 20, std: float = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function simulates the Belief Propagation (BP) algorithm for estimating the offset and skew in a wireless
        sensor network. The network consists of several nodes, each with its own clock, and factors, which represent
        communication links between the nodes.
        The Grand Master node serves as a reference for the other nodes. The BP algorithm is used to estimate the offset
        and skew of each node based on the timestamps exchanged between the nodes via the factors.

        Args:
        - sim_iter (int): Number of iterations to run the simulation
        - low (int): Lower bound for initial offsets
        - high (int): Upper bound for initial offsets
        - Nodes (list): List of nodes in the network
        - Factors (list): List of factors (communication links) between nodes
        - GrandMaster (str): The Grand Master node
        - L_bp_iter (int): Number of BP iterations
        - num_ex (int): Number of timestamps to generate per iteration
        - std (float, optional): Standard deviation of timestamps. Default is 8
        - KF_flag (int, optional): Flag to indicate if Kalman Filter is to be applied. Default is 0
        - KF_nodes (list, optional): List of nodes to apply Kalman Filter. Default is empty

        Returns:
        - the_est_fin : estimated offset for each node
        - skw_est_fin : estimated skew for each node
        - the_std_est_fin : standard deviation of the offset estimates
        - skw_std_est_fin : standard deviation of the skew estimates
        """
        # defining the variable vectors
        the_est_fin = np.zeros((L_bp_iter, len(self.Nodes + self.KF_nodes)))
        skw_est_fin = np.zeros((L_bp_iter, len(self.Nodes + self.KF_nodes)))
        the_std_est_fin = np.zeros((L_bp_iter, len(self.Nodes + self.KF_nodes)))
        skw_std_est_fin = np.zeros((L_bp_iter, len(self.Nodes + self.KF_nodes)))
        F_prior_mu = {}
        F_prior_std = {}
        gamma = {}
        the = {}
        skw_std = 1e-2
        # defining factors corresponding to the prior knowledge on GM and nodes
        # Beta_i = [1/alpha_i , theta_i/alpha_i]
        for n in self.Nodes:
            if GrandMaster == n.split('_')[1]:
                temp = str('F_') + n.split('_')[1]
                F_prior_mu.update({temp: np.array([[1, 0]]).T})
                F_prior_std.update({temp: np.array([[1e-12, 0], [0, 1e-1]])})
            else:
                temp = str('F_') + n.split('_')[1]
                F_prior_mu.update({temp: np.array([[1, 0]]).T})
                F_prior_std.update({temp: np.array([[1e-3, 0], [0, 1e+8]])})
        relative_offset = 0
        for _ in np.arange(sim_iter):
            # print(u)
            for nd in self.Nodes + self.KF_nodes:
                node_ind = nd.split('_')[1]
                gamma.update({str('gamma_') + node_ind: 1 + np.random.normal(0, skw_std, 1)})
                the.update({str('the_') + node_ind: np.random.randint(low, high, 1)})
            gamma['gamma_' + str(GrandMaster)] = 1
            the['the_' + str(GrandMaster)] = 0

            # Constructing the measurement matrices for Factor Graph
            measure_vec = {}
            for fac in self.Factors:
                node_i = str(list(fac.split('_')[1])[0])
                node_j = str(''.join(list(fac.split('_')[1])[1::]))
                the_i = the[str('the_') + node_i]
                the_j = the[str('the_') + node_j]
                gamma_i = gamma[str('gamma_') + node_i]
                gamma_j = gamma[str('gamma_') + node_j]
                C_stamp = self.timestamp_exchange(the_i, the_j, gamma_i, gamma_j, num_ex, std, asym=True)
                c_ji = (C_stamp[1, ::] + C_stamp[3, ::]) / 2 + C_stamp[4, ::]
                c_ij = (C_stamp[0, ::] + C_stamp[2, ::]) / 2 + C_stamp[5, ::]
                apend_vec = -2 * np.ones((1, num_ex))
                measure_vec.update({str('A_') + node_j + node_i: np.append([c_ji], apend_vec, axis=0).T})
                measure_vec.update({str('A_') + node_i + node_j: -np.append([c_ij], apend_vec, axis=0).T})

            # Time stamps for the KF links
            ts = {}
            if self.syncType == 'hybrid':
                for nd in self.KF_nodes:
                    node_i = nd.split('_')[1][0]
                    node_j = nd.split('_')[1][1]
                    c_iij = self.timestamp_exchange(the[str('the_') + node_i], the[str('the_') + node_i + node_j],
                                         gamma[str('gamma_') + node_i], gamma[str('gamma_') + node_i + node_j], num_ex,
                                         std)
                    ts.update({str('ts_') + node_i + node_i + node_j: c_iij})

            std_msg = {}
            mu_msg = {}
            std_msg_fin = {}
            mu_msg_fin = {}
            cons_init_mu = 1e-10
            cons_init_std = 1e+10
            for fac in self.Factors:
                node_i = str(list(fac.split('_')[1])[0])
                node_j = str(''.join(list(fac.split('_')[1])[1::]))
                std_msg.update({node_i + str('to') + node_j: cons_init_std * np.eye(2)})
                std_msg.update({node_j + str('to') + node_i: cons_init_std * np.eye(2)})
                mu_msg.update({node_i + str('to') + node_j: cons_init_mu * np.array([[1, 1]]).T})
                mu_msg.update({node_j + str('to') + node_i: cons_init_mu * np.array([[1, 1]]).T})
                std_msg_fin.update({node_i + str('to') + node_j: cons_init_std * np.eye(2)})
                std_msg_fin.update({node_j + str('to') + node_i: cons_init_std * np.eye(2)})
                mu_msg_fin.update({node_i + str('to') + node_j: cons_init_mu * np.array([[1, 1]]).T})
                mu_msg_fin.update({node_j + str('to') + node_i: cons_init_mu * np.array([[1, 1]]).T})

            temp_the_est = {}
            temp_skew_est = {}
            relative_offset = 0
            for l in np.arange(L_bp_iter):

                # \delta_{i->j} messages
                for out in list(std_msg):
                    # taking the priors into account
                    calc_temp_std = inv(F_prior_std[str('F_') + out.split('to')[0]])
                    calc_temp_mu = np.dot(inv(F_prior_std[str('F_') + out.split('to')[0]]),
                                          F_prior_mu[str('F_') + out.split('to')[0]])

                    # removing the message \delta_{j->j} when calculating incoming messages of i
                    in_msg = list(std_msg)
                    in_msg.remove(out)

                    # calculating the incoming messages of node i
                    for pair in in_msg:
                        if pair.split('to')[1] == out.split('to')[0]:
                            calc_temp_std = calc_temp_std + inv(std_msg[pair])
                            calc_temp_mu = calc_temp_mu + np.dot(inv(std_msg[pair]), mu_msg[pair])
                    Aji = measure_vec['A_' + out.split('to')[0] + out.split('to')[1]]
                    Aij = measure_vec['A_' + out.split('to')[1] + out.split('to')[0]]

                    # updating mean and covariance based on iterative formulas
                    calc_temp = std ** 2 * np.eye(num_ex) + multi_dot([Aji, inv(calc_temp_std), Aji.T])
                    std_msg_fin[out] = inv(multi_dot([Aij.T, inv(calc_temp), Aij]))
                    mu_msg_fin[out] = -multi_dot(
                        [std_msg_fin[out], Aij.T, inv(calc_temp), Aji, inv(calc_temp_std), calc_temp_mu])
                for out in list(std_msg):
                    std_msg[out] = std_msg_fin[out]
                    mu_msg[out] = mu_msg_fin[out]

                # calculating the belief of each node
                nd = -1
                for k in self.Nodes:
                    nd += 1
                    node = str(k.split('_')[1])
                    est_std = inv(F_prior_std[str('F_') + node])
                    est_mu = np.dot(inv(F_prior_std[str('F_') + node]), F_prior_mu[str('F_') + node])
                    for fac in list(mu_msg):
                        if fac.split('to')[1] == node:
                            est_std = est_std + inv(std_msg[fac])
                            est_mu = est_mu + np.dot(inv(std_msg[fac]), mu_msg[fac])
                    est_mu = np.dot(inv(est_std), est_mu)
                    # print(str('(')+node+str(')'),the[str('the_')+node], gamma[str('gamma_')+node], 1/est_mu[0], est_mu[1]/est_mu[0])
                    temp_the_est.update({str('the_') + node: the[str('the_') + node] - est_mu[1] / est_mu[0]})
                    temp_skew_est.update({str('gamma_') + node: gamma[str('gamma_') + node] - 1 / est_mu[0]})
                    the_est_fin[l, nd] = the[str('the_') + node] - est_mu[1] / est_mu[0]
                    the_std_est_fin[l, nd] = the_std_est_fin[l, nd] + 1 / (est_std[1, 1] * est_std[0, 0])
                    skw_std_est_fin[l, nd] = skw_std_est_fin[l, nd] + 1 / est_std[0, 0]
                if self.syncType == 'hybrid':
                    nd = len(self.Nodes) - 1
                    for kf_nd in self.KF_nodes:
                        nd += 1
                        node_i = str(list(kf_nd.split('_')[1])[0])
                        node_j = str(list(kf_nd.split('_')[1])[1])

                        mu, std_est = self.brf(0.0, the[str('the_') + node_i + node_j],
                                             1, gamma[str('gamma_') + node_i + node_j], l, num_ex, std)
                        # the_est_fin[l,nd] = the_est_fin[l,nd] + (mu[0]-temp_the_est[str('the_')+node_i]
                        #                                                      - the[str('the_')+node_i+node_j])**2
                        the_est_fin[l, nd] = (the[str('the_') + node_i + node_j] - mu[0] - mu[1] * temp_the_est[
                            str('the_') + node_i])

                        skw_est_fin[l, nd] = (gamma[str('gamma_') + node_i + node_j] - mu[1] - mu[1] * temp_skew_est[
                            str('gamma_') + node_i])

                        # skw_est_fin[l, nd] = skw_est_fin[l, nd] \
                        #                      + (gamma[str('gamma_')+node_i+node_j]/(1+temp_skew_est[str('gamma_')+node_i])
                        #                       - mu[1] + temp_skew_est[str('gamma_')+node_i])**2
                        the_std_est_fin[l, nd] = the_std_est_fin[l, nd] + std_est[0]
                        skw_std_est_fin[l, nd] = skw_std_est_fin[l, nd] + std_est[1]
            relative_offset = relative_offset + (the_est_fin[::, -3] - the_est_fin[::, -4])

        return relative_offset[4], the_est_fin[l, [-3, -4]], skw_est_fin[l, [-3, -4]]

    @staticmethod
    def timestamp_exchange(the_i: float, the_j: float, gamma_i: float, gamma_j: float, num_ex: int, std: float,
                           asym: bool = False) -> np.ndarray:
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
        d_ij = np.random.randint(200, 300)
        T = (1 / 128) * 1e+9
        waiting_time0 = 0.999 * T
        waiting_time = 2 * d_ij
        iter = num_ex
        if asym:
            C_stamp = np.zeros((6, iter))
            for i in range(iter):
                t = i*T
                # egress and ingress time (are generally random)
                T_n0 = np.abs(np.random.normal(30, sigma_eg))
                T_n = np.abs(np.random.normal(30, sigma_eg))
                R_n = np.abs(np.random.normal(30, sigma_ing))
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
                T_n = np.random.normal(30, sigma_eg)
                R_n = np.random.normal(30, sigma_ing)
                C_stamp[0, i] = gamma_i * (t + waiting_time) + the_i
                C_stamp[1, i] = gamma_j * (t + waiting_time + d_ij + T_n) + the_j
                C_stamp[2, i] = gamma_j * (t + waiting_time + d_ij + T_n + waiting_time) + the_j
                C_stamp[3, i] = gamma_i * (t + waiting_time + d_ij + T_n + waiting_time + d_ij + R_n) + the_i
        return C_stamp