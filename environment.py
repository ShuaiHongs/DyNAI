import networkx as nx
import numpy as np
import utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nx_adj_to_dense_tensor(G, nodelist, weight=None, transpose=False, device=device):
    """
    将 networkx 邻接矩阵转成 dense float tensor
    """
    adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight)
    dense = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)

    if transpose:
        dense = dense.t()

    return dense


class Simulation(object):
    def __init__(self, dataset, num_of_actions, budget, graph_hold_steps=2):
        """
        Parameters
        ----------
        dataset : str
            数据集路径
        num_of_actions : int
            动作数
        budget : float
            总预算
        graph_hold_steps : int, default=2
            每个 snapshot 保持多少个 environment step 再切换到下一个 snapshot
            - 1: 和原来完全一致
            - 2/3: 图变化更平缓，通常更利于学习
        """
        self.graphs, self.all_nodes, self.snapshot_list = utils.import_users_data_from_csv(dataset)

        self.num_of_actions = num_of_actions
        self.budget = budget

        self.user_id = list(self.all_nodes)
        self.user_index2id = {i: user for i, user in enumerate(self.user_id)}
        self.user_id2index = {user: i for i, user in enumerate(self.user_id)}
        self.num_of_user = len(self.user_id)

        self.t = 0
        self.G = None

        self.action_matrix = None
        self.feature_matrix = None
        self.action_one_hot_matrix = None

        self.last_active_rate = 0.0
        self.remaining_budget = None
        self.initial_budget = budget

        # 图保持步数
        self.graph_hold_steps = max(1, int(graph_hold_steps))
        self.step_in_current_snapshot = 0

        self._update_graph()

    def _update_graph(self):
        self.G = self.graphs[self.snapshot_list[self.t]]

        self.adjacency_matrix_out = nx_adj_to_dense_tensor(
            self.G, self.user_id, weight=None, transpose=False, device=device
        )
        self.adjacency_matrix_in = nx_adj_to_dense_tensor(
            self.G, self.user_id, weight=None, transpose=True, device=device
        )

        self.preference_matrix = torch.FloatTensor(
            [self.G.nodes[user]['Preference'] for user in self.user_id]
        ).to(device)

        self.weight_matrix = nx_adj_to_dense_tensor(
            self.G, self.user_id, weight="Weight", transpose=True, device=device
        )

        self.user_in_degree = torch.sum(self.adjacency_matrix_in, dim=1)
        self.user_out_degree = torch.sum(self.adjacency_matrix_out, dim=1)

    def reset(self):
        self.t = 0
        self.step_in_current_snapshot = 0
        self._update_graph()

        self.remaining_budget = torch.tensor([self.budget], dtype=torch.float32, device=device)

        self.action_matrix = torch.argmax(self.preference_matrix, dim=1)
        self.action_one_hot_matrix = torch.eye(self.num_of_actions, device=device)[self.action_matrix]

        incentive = torch.zeros((self.num_of_user, 1), device=device)
        remaining_ratio = (self.remaining_budget / self.initial_budget).view(1, 1).expand(self.num_of_user, 1)

        per_user_feature = torch.cat([incentive, self.action_one_hot_matrix, remaining_ratio], dim=1)
        self.feature_matrix = per_user_feature.flatten()

        active_num = torch.sum(self.action_matrix == 0).item()
        self.last_active_rate = active_num / self.num_of_user

        return (
            self.feature_matrix.cpu().numpy(),
            np.array([self.last_active_rate]),
            np.array([active_num])
        )

    def step(self, inc_action):
        inc_action = torch.tensor(inc_action, dtype=torch.float32, device=device).view(-1)

        # 当前 snapshot 下的影响计算
        influence = self.weight_matrix @ self.action_one_hot_matrix
        utility_matrix = self.preference_matrix + influence
        utility_dif = utility_matrix[:, 0] - torch.max(utility_matrix, dim=1)[0]

        incentives = (inc_action + 1) / 2
        remaining_budget = self.remaining_budget.clone()

        spent_incentives = torch.where(
            incentives + utility_dif >= 0,
            incentives,
            torch.zeros_like(incentives)
        )

        cum_incentives = torch.cumsum(spent_incentives, dim=0)
        mask = cum_incentives <= remaining_budget
        spent_incentives_ = torch.where(mask, spent_incentives, torch.zeros_like(spent_incentives))

        remaining_budget -= torch.sum(spent_incentives_)

        if remaining_budget.item() > 0:
            unsatisfied = (mask == 0).nonzero(as_tuple=True)[0]
            if len(unsatisfied) > 0:
                first_unsatisfied = unsatisfied[0]
                if utility_dif[first_unsatisfied] + remaining_budget >= 0:
                    spent_incentives_[first_unsatisfied] = remaining_budget
                    remaining_budget = torch.tensor([0.0], device=device)

        utility_matrix[:, 0] += spent_incentives_
        self.action_matrix = torch.argmax(utility_matrix, dim=1)
        self.action_one_hot_matrix = torch.eye(self.num_of_actions, device=device)[self.action_matrix]

        remaining_ratio = (remaining_budget / self.initial_budget).view(1, 1).expand(self.num_of_user, 1)
        per_user_feature = torch.cat([spent_incentives_.unsqueeze(1), self.action_one_hot_matrix, remaining_ratio], dim=1)
        self.feature_matrix = per_user_feature.flatten()

        active_num = torch.sum(self.action_matrix == 0).item()
        rate = active_num / self.num_of_user

        rewards = self.reward(spent_incentives_, rate, remaining_budget)

        self.last_active_rate = rate
        self.remaining_budget = remaining_budget

        done = false_done = False

        # -------------------------------------------------------
        # 不是每步都切图，而是每 graph_hold_steps 步切一次
        # -------------------------------------------------------
        self.step_in_current_snapshot += 1

        if self.step_in_current_snapshot >= self.graph_hold_steps:
            self.step_in_current_snapshot = 0

            if self.t + 1 < len(self.snapshot_list):
                self.t += 1
                self._update_graph()
            else:
                # 已经是最后一个 snapshot，并且当前窗口结束，episode 结束
                done = True
        else:
            # 还没到切图时机，但如果已经是最后一个 snapshot，也允许继续在当前图上执行
            # 直到这个 snapshot 的 hold window 用完才 done
            if self.t == len(self.snapshot_list) - 1 and self.step_in_current_snapshot == 0:
                done = True

        return (
            self.feature_matrix.cpu().numpy(),
            rewards.cpu().numpy(),
            np.array([rate]),
            np.array([active_num]),
            remaining_budget.cpu().numpy(),
            done
        )

    def reward(self, spent_incentives, rate, remaining_budget, adj_delta=None):
        """
        更适合 DyNAI 的 reward：
        - 以激活率为主
        - 鼓励比上一时刻提升
        - 轻微鼓励保留预算
        - 轻微惩罚花费
        """
        delta_active = rate - self.last_active_rate
        total_spent = torch.sum(spent_incentives).item()
        remaining_ratio = (remaining_budget / self.initial_budget).item()

        active_bonus = 6.0 * rate
        improvement_bonus = 4.0 * delta_active
        budget_balance_bonus = 0.5 * remaining_ratio
        spend_penalty = 0.03 * total_spent

        reward_val = active_bonus + improvement_bonus + budget_balance_bonus - spend_penalty


        if adj_delta is not None:
            if torch.is_tensor(adj_delta):
                graph_change = torch.mean(torch.abs(adj_delta)).item()
            else:
                graph_change = float(np.mean(np.abs(adj_delta)))
            reward_val += 0.2 * graph_change * max(delta_active, 0.0)

        return torch.tensor(reward_val, dtype=torch.float32, device=device)

    def sample_action(self):
        return torch.clip(
            torch.normal(0, 1, size=(self.num_of_user,)),
            -1, 1
        ).to(device)