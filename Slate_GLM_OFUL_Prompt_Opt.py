import numpy as np
from optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from utils import sigmoid, dsigmoid, weighted_norm, gaussian_sample_ellipsoid
from datetime import datetime
from time import time

class Slate_GLM_OFUL_Prompt_Opt():
    def __init__(self, num_examples , example_pool_size , embedding_dim , failure_level , param_norm_ub , start_with , data_path , horizon , repeat_examples):
        self.slot_count = num_examples
        self.item_count = example_pool_size
        self.dim_per_action = 3 * embedding_dim      # to acccount for the query embedding and label embedding
        self.dim = self.slot_count * self.dim_per_action
        self.failure_level = failure_level
        self.param_norm_ub = param_norm_ub
        self.l2reg = 5
        self.horizon = horizon
        self.repeat_examples = repeat_examples
        if start_with < 0:
            self.vtilde_matrix = self.l2reg * np.eye(self.dim)
            self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
            self.v_matrices_inv = [(1 / self.l2reg) * np.eye(self.dim_per_action) for _ in range(self.slot_count)]
            self.v_matrices = [self.l2reg * np.eye(self.dim_per_action) for _ in range(self.slot_count)]
            self.theta = np.array([np.random.random()*2-1 for i in range(self.dim)])
            self.theta /= np.linalg.norm(self.theta)
            self.conf_radius = 0
            self.cum_loss = 0
            self.ctr = 0
        else:
            folder_name = data_path + "/parameters_{}".format(start_with)
            print("Loading parameters from {}".format(folder_name))
            self.vtilde_matrix = np.load(folder_name + "/vtilde_matrix.npy")
            self.vtilde_matrix_inv = np.load(folder_name + "/vtilde_matrix_inv.npy")
            self.v_matrices = np.load(folder_name + "/v_matrices.npy")
            self.v_matrices_inv = np.load(folder_name + "/v_matrices_inv.npy")
            self.theta = np.load(folder_name + "/theta.npy")
            self.conf_radius = np.load(folder_name + "/conf_radius.npy")
            self.cum_loss = np.load(folder_name + "/cum_loss.npy")
            self.ctr = np.load(folder_name + "/ctr.npy")

    def update_parameters(self, picked_arms, reward):
        '''
        function to update the parameters after each run of the alg
        '''
        
        arm = np.hstack([*picked_arms])

        # linearly increasing precision between 0.1 and 0.01
        precision = -0.09*self.ctr/(self.horizon-1) + (0.1 + 0.09/(self.horizon-1)) 
        # compute new estimate theta
        self.theta = np.real_if_close(fit_online_logistic_estimate(arm=arm,
                                                                reward=reward,
                                                                current_estimate=self.theta,
                                                                vtilde_matrix=self.vtilde_matrix,
                                                                vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                constraint_set_radius=self.param_norm_ub,
                                                                diameter=self.param_norm_ub,
                                                                precision=precision))
        # compute theta_bar (needed for data-dependent conf. width)
        theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(arm=arm,
                                                                    current_estimate=self.theta,
                                                                    vtilde_matrix=self.vtilde_matrix,
                                                                    vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                    constraint_set_radius=self.param_norm_ub,
                                                                    diameter=self.param_norm_ub,
                                                                    precision=precision))
        disc_norm = np.clip(weighted_norm(self.theta-theta_bar, self.vtilde_matrix), 0, np.inf)

        # update matrices
        sensitivity = dsigmoid(np.dot(self.theta, arm))
        self.vtilde_matrix += sensitivity * np.outer(arm, arm)
        self.vtilde_matrix_inv += - sensitivity * np.dot(self.vtilde_matrix_inv,
                                                        np.dot(np.outer(arm, arm), self.vtilde_matrix_inv)) / (
                                          1 + sensitivity * np.dot(arm, np.dot(self.vtilde_matrix_inv, arm)))
        # sensitivity check
        sensitivity_bar = dsigmoid(np.dot(theta_bar, arm))
        if sensitivity_bar / sensitivity > 2:
            msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            raise ValueError(msg)

        # update sum of losses
        coeff_theta = sigmoid(np.dot(self.theta, arm))
        loss_theta = -reward * np.log(coeff_theta) - (1-reward) * np.log(1-coeff_theta)
        coeff_bar = sigmoid(np.dot(theta_bar, arm))
        loss_theta_bar = -reward * np.log(coeff_bar) - (1-reward) * np.log(1-coeff_bar)
        self.cum_loss += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm

        # extract the slotwise arm without the zeros
        slot_wise_arms = picked_arms

        # create temp arms which are sqrt(sigmoid(arm*theta))
        temp_slot_wise_arms = [dsigmoid(np.sum(self.theta * arm))**0.5 * a for a in slot_wise_arms]
        
        # update the slotwise parameters
        for idx , arm in enumerate(temp_slot_wise_arms):
            self.v_matrices[idx] += np.outer(arm, arm)
            self.v_matrices_inv[idx] = self.sherman_morrison_update(self.v_matrices_inv[idx] , arm , arm)

    def pull(self, arm_set , seperate_pools = False):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        self.update_ucb_bonus()
        pulled_arms_indices = self.slotwise_argmax(self.compute_optimistic_reward , arm_set , seperate_pools)  
        self.ctr += 1
        return pulled_arms_indices

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (a more precise version of Thm3 in ECOLog paper, refined for the no-warm up alg)
        """
        gamma = np.sqrt(self.l2reg) / 2 + 2 * np.log(
            2 * np.sqrt(1 + self.ctr / (4 * self.l2reg)) / self.failure_level) / np.sqrt(self.l2reg)
        res_square = 2*self.l2reg*self.param_norm_ub**2 + (1+self.param_norm_ub)**2*gamma + self.cum_loss
        res_square = max(0 , res_square)
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm , slot_idx):
        """
        Returns prediction + exploration_bonus for arm.
        """
        norm = weighted_norm(arm, self.v_matrices_inv[slot_idx])
        local_theta = self.theta[slot_idx * self.dim_per_action : (slot_idx + 1) * self.dim_per_action]
        pred_reward = np.dot(local_theta , arm)
        bonus = self.conf_radius * norm
        return pred_reward + bonus
    
    def sherman_morrison_update(self , v_inv , vec1 , vec2):
        '''
        implements the sherman morrison update for inverse of rank 1 additions
        '''
        return v_inv - v_inv@np.outer(vec1 , vec2)@v_inv/(1 + np.dot(vec1, v_inv@vec2))

    def slotwise_argmax(self , fun , arm_set , seperate_pools = False):
        '''
        returns the best arm in each set maximizing fun
        '''
        picked_actions_indices = []
        for slot in range(self.slot_count):
            # if seperate pools each slot has its own set of arms
            slot_arms = arm_set[slot] if seperate_pools else arm_set
            
            slot_values = [(i,fun(a , slot)) for i,a in enumerate(slot_arms)]
            sorted_slot_values = sorted(slot_values , key = lambda x: x[1] , reverse = True)
            
            # if seperate pools then no worry about repetition of arms
            if seperate_pools:
                picked_actions_indices.append(sorted_slot_values[0][0])
            else:
                if self.repeat_examples:
                    picked_actions_indices.append(sorted_slot_values[0][0])
                else:
                    # ensure examples donot repeat over slots
                    for i in range(len(sorted_slot_values)):
                        if sorted_slot_values[i][0] not in picked_actions_indices:
                            picked_actions_indices.append(sorted_slot_values[i][0])
                            break

        return picked_actions_indices
        