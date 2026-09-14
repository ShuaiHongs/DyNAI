import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 工具函数
# ============================================================
def to_dense_adj(adj):
    if adj is None:
        return None

    if torch.is_tensor(adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        return adj.to(device=device, dtype=torch.float32)

    return torch.tensor(adj, dtype=torch.float32, device=device)


def dense_row_normalize(adj):
    if adj is None:
        return None

    if adj.dim() == 2:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return adj / deg
    elif adj.dim() == 3:
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1e-12)
        return adj / deg
    else:
        raise ValueError(f"adj must be 2D or 3D, got {adj.shape}")


def ensure_tensor(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device=device, dtype=torch.float32)
    return x


# ============================================================
# Dense GCN Layer
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, features, adj):
        if features.dim() == 2:
            features = features.unsqueeze(0)

        if adj is None:
            raise ValueError("adj 不能为空")

        adj = ensure_tensor(adj)
        if adj.is_sparse:
            adj = adj.to_dense()

        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        # 自动广播 batch
        if adj.size(0) == 1 and features.size(0) > 1:
            adj = adj.expand(features.size(0), -1, -1)
        if features.size(0) == 1 and adj.size(0) > 1:
            features = features.expand(adj.size(0), -1, -1)

        h = self.dropout(features)
        h = self.linear(h)
        h = torch.bmm(adj, h)
        h = self.ln(F.relu(h))
        return h


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, dropout)

    def forward(self, features, adj):
        h = self.gcn1(features, adj)
        h = self.gcn2(h, adj)
        return h


# ============================================================
# Actor
# ============================================================
class Actor(nn.Module):
    def __init__(self, input_dim, g_hidden_dim, hidden_dim, action_dim, max_action=1.0):
        super().__init__()
        self.gcn_out = GCNEncoder(input_dim, g_hidden_dim)
        self.gcn_in = GCNEncoder(input_dim, g_hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(g_hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, features, adj_out, adj_in):
        if features.dim() == 2:
            features = features.unsqueeze(0)

        out_emb = self.gcn_out(features, adj_out)
        in_emb = self.gcn_in(features, adj_in)

        fused = torch.cat([out_emb, in_emb], dim=-1)
        action = self.max_action * self.net(fused)
        return action


# ============================================================
# Critic
# ============================================================
class Critic(nn.Module):
    def __init__(self, input_dim, g_hidden_dim, hidden_dim, action_dim):
        super().__init__()

        fused_dim = g_hidden_dim * 2 + action_dim

        self.gcn_out_1 = GCNEncoder(input_dim, g_hidden_dim)
        self.gcn_in_1 = GCNEncoder(input_dim, g_hidden_dim)
        self.q1_net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.gcn_out_2 = GCNEncoder(input_dim, g_hidden_dim)
        self.gcn_in_2 = GCNEncoder(input_dim, g_hidden_dim)
        self.q2_net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _encode(self, features, adj_out, adj_in, action, gcn_out, gcn_in):
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if action.dim() == 2:
            action = action.unsqueeze(0)

        out_emb = gcn_out(features, adj_out)
        in_emb = gcn_in(features, adj_in)
        fused = torch.cat([out_emb, in_emb, action], dim=-1)
        return fused

    def forward(self, features, adj_out, adj_in, action):
        sa1 = self._encode(features, adj_out, adj_in, action, self.gcn_out_1, self.gcn_in_1)
        q1 = self.q1_net(sa1).mean(dim=1)

        sa2 = self._encode(features, adj_out, adj_in, action, self.gcn_out_2, self.gcn_in_2)
        q2 = self.q2_net(sa2).mean(dim=1)

        return q1, q2

    def Q1(self, features, adj_out, adj_in, action):
        sa1 = self._encode(features, adj_out, adj_in, action, self.gcn_out_1, self.gcn_in_1)
        return self.q1_net(sa1).mean(dim=1)


# ============================================================
# ============================================================
class GAC(nn.Module):
    def __init__(
        self,
        state_dim,
        g_hidden_dim,
        fc_hidden_dim,
        action_dim,
        max_action,
        discount,
        tau,
        policy_noise,
        noise_clip,
        policy_freq,
        actor_lr,
        critic_lr,
        num_of_users,
        history_len=5,
        num_layer=2,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.per_user_dim = int(state_dim / num_of_users)
        self.num_of_users = num_of_users

        self.actor = Actor(
            input_dim=self.per_user_dim,
            g_hidden_dim=g_hidden_dim,
            hidden_dim=fc_hidden_dim,
            action_dim=action_dim,
            max_action=max_action
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

        self.critic = Critic(
            input_dim=self.per_user_dim,
            g_hidden_dim=g_hidden_dim,
            hidden_dim=fc_hidden_dim,
            action_dim=action_dim
        ).to(device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.last_actor_loss = float("nan")

        self.adjs_out = None
        self.adjs_in = None

    def set_adjs(self, adjs_out, adjs_in):
        self.adjs_out = dense_row_normalize(to_dense_adj(adjs_out))
        self.adjs_in = dense_row_normalize(to_dense_adj(adjs_in))

    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.float().to(device)

        if state.dim() == 1:
            state = state.view(1, self.num_of_users, self.per_user_dim)
        elif state.dim() == 2:
            state = state.unsqueeze(0)

        adjs_out = self.adjs_out
        adjs_in = self.adjs_in
        if adjs_out.dim() == 2:
            adjs_out = adjs_out.unsqueeze(0)
        if adjs_in.dim() == 2:
            adjs_in = adjs_in.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state, adjs_out, adjs_in)

        return action.squeeze(0).flatten()

    def soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self, replay_buffer, batch_size):
        self.total_it += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, with_aux=False)

        states = states.view(batch_size, self.num_of_users, self.per_user_dim)
        next_states = next_states.view(batch_size, self.num_of_users, self.per_user_dim)
        actions = actions.view(batch_size, self.num_of_users, -1)
        rewards = rewards.view(batch_size, 1)
        dones = dones.view(batch_size, 1)

        if actions.size(-1) != 1:
            actions = actions[..., :1]

        adjs_out = self.adjs_out
        adjs_in = self.adjs_in

        if adjs_out.dim() == 2:
            adjs_out = adjs_out.unsqueeze(0)
        if adjs_in.dim() == 2:
            adjs_in = adjs_in.unsqueeze(0)

        if adjs_out.size(0) == 1 and states.size(0) > 1:
            adjs_out = adjs_out.expand(states.size(0), -1, -1)
        if adjs_in.size(0) == 1 and states.size(0) > 1:
            adjs_in = adjs_in.expand(states.size(0), -1, -1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_states, adjs_out, adjs_in)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_states, adjs_out, adjs_in, next_action)
            target_Q = rewards + (1.0 - dones) * self.discount * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(states, adjs_out, adjs_in, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_update = False
        if self.total_it % self.policy_freq == 0:
            pred_action = self.actor(states, adjs_out, adjs_in)
            actor_loss = -self.critic.Q1(states, adjs_out, adjs_in, pred_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.soft_update()
            self.last_actor_loss = actor_loss.item()
            actor_update = True

        return critic_loss.item(), self.last_actor_loss, actor_update

