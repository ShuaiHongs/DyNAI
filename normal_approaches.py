# normal_approaches.py
import numpy as np
import math
import random

# =====================================================
# 1. 无激励方法
# =====================================================
class NoIn(object):
    def __init__(self, G):
        self.G = G
        self.name = "No Incentive"

    def allocate_incentives(self, user_id, budget):
        return 0.0

    def update(self, *args, **kwargs):
        pass


# =====================================================
# 2. 随机方法
# =====================================================
class Random(object):
    def __init__(self, G):
        self.G = G
        self.name = "Random"

    def allocate_incentives(self, user_id, remaining_budget):
        if user_id not in self.G or remaining_budget <= 0:
            return 0.0
        return float(random.random())

    def update(self, *args, **kwargs):
        pass


# =====================================================
# 3. 平均分配
# =====================================================
class Uniform(object):
    def __init__(self, G):
        self.G = G
        self.name = "Uniform"

    def allocate_incentives(self, user_id, budget):
        if user_id not in self.G:
            return 0.0

        num_nodes = len(self.G.nodes())
        return float(budget / num_nodes) if num_nodes > 0 else 0.0

    def update(self, *args, **kwargs):
        pass


# =====================================================
# 4. K-MAB（可学习版）
# =====================================================
class kmab(object):
    def __init__(self, G, arm_num=50):
        self.G = G
        self.arm_num = arm_num

        self.price = np.linspace(0.01, 1.0, arm_num)
        self.P = np.zeros(arm_num)
        self.counts = np.ones(arm_num)

        self.price_index = 0
        self.name = "K-MAB"

    def allocate_incentives(self, user_id, remaining_budget):
        if remaining_budget <= 0 or user_id not in self.G:
            return 0.0

        # UCB-like exploration
        ucb = self.P + np.sqrt(1 / self.counts)

        self.price_index = int(np.argmax(ucb))

        return float(min(self.price[self.price_index], remaining_budget))

    def update(self, user_id, reward):
        idx = self.price_index

        self.counts[idx] += 1
        self.P[idx] += (reward - self.P[idx]) / self.counts[idx]


# =====================================================
# 5. DGIA（动态图激励算法）
# =====================================================
class DGIA(object):
    def __init__(self, G):
        self.G = G

        self.omega = {}
        self.incentive_degree = {}
        self.influence_degree = {}
        self.influence = {}

        for uid in self.G.nodes():
            self._init_node_params(uid)

        self.beta = 0.2
        self.name = "DGIA-IPE"

    def _init_node_params(self, uid):
        if uid not in self.incentive_degree:
            pref = [0.5, 0.5]

            if uid in self.G and 'Preference' in self.G.nodes[uid]:
                pref = self.G.nodes[uid]['Preference']

            total = np.sum(pref) if np.sum(pref) > 0 else 1.0

            self.omega[uid] = pref[0] / total
            self.incentive_degree[uid] = 0.5
            self.influence_degree[uid] = 0.0
            self.influence[uid] = {}

    def _check_new_user(self, uid):
        if uid not in self.incentive_degree:
            self._init_node_params(uid)

    def allocate_incentives(self, user_id, remaining_budget):
        if remaining_budget <= 0 or user_id not in self.G:
            return 0.0

        self._check_new_user(user_id)

        node_data = self.G.nodes[user_id]
        preference = node_data.get('Preference', [0.5, 0.5])

        pre_dif = float(np.max(preference) - preference[0])

        idegree = float(self.influence_degree.get(user_id, 0.0))
        incen_degree = float(self.incentive_degree.get(user_id, 0.5))

        reward = (1 - incen_degree) * (pre_dif + 2.0 * idegree)

        return float(min(reward, remaining_budget))

    def update_incentive_degree(self, user_id, action):
        self._check_new_user(user_id)

        value = self.incentive_degree[user_id]
        omega = self.omega[user_id]

        if action == 0:
            self.incentive_degree[user_id] = value / (value + omega * (1 - value) + 1e-6)
        else:
            self.incentive_degree[user_id] = value * 0.7

    def estimate_influence(self, user_id, user_action, time):
        self._check_new_user(user_id)

        for uid in self.G.nodes():
            if uid == user_id:
                continue

            self._check_new_user(uid)

            node_data = self.G.nodes[uid]

            if 'Action' not in node_data:
                continue

            actions_ = node_data['Action']

            if time >= len(actions_):
                continue

            x = self.influence[uid].get(user_id, 0.0)

            if user_action == actions_[time]:
                x += self.beta * (1 - x)
            else:
                x -= self.beta * x

            self.influence[uid][user_id] = float(np.clip(x, 0.0, 1.0))

    def calculate_influence_degree(self):
        n = len(self.G.nodes())
        if n == 0:
            return

        for k in self.influence:
            vals = list(self.influence[k].values())
            self.influence_degree[k] = float(np.mean(vals)) if vals else 0.0

    def update(self, user_id, action, time):
        action = int(action > 0)

        self.update_incentive_degree(user_id, action)
        self.estimate_influence(user_id, action, time)
        self.calculate_influence_degree()