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


def expand_batch(x, batch_size):
    if x is None:
        return None
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.size(0) == 1 and batch_size > 1:
        x = x.expand(batch_size, *x.shape[1:])
    return x


# ============================================================
# Temporal Encoder
# ============================================================
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=40, nhead=4, dropout=0.1, max_len=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.ln = nn.LayerNorm(hidden_dim)

        self.mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, history_features):
        if history_features is None:
            return None

        history_features = ensure_tensor(history_features)

        if history_features.dim() == 3:
            history_features = history_features.unsqueeze(0)

        if history_features.dim() != 4:
            raise ValueError(
                f"history_features must be [B,S,N,D] or [S,N,D], got {history_features.shape}"
            )

        b, s, n, d = history_features.shape
        x = history_features.permute(0, 2, 1, 3).reshape(b * n, s, d)
        x = self.input_proj(x)

        pos = self.pos_embedding[:, :s, :].expand(b * n, -1, -1)
        x = x + pos

        out = self.transformer(x)

        last_emb = out[:, -1, :]
        mean_emb = out.mean(dim=1)

        emb = self.mix(torch.cat([last_emb, mean_emb], dim=-1))
        emb = self.ln(emb)
        return emb.view(b, n, self.hidden_dim)


# ============================================================
# Graph Evolution Encoder
# ============================================================
class GraphEvolutionEncoder(nn.Module):
    def __init__(self, n_users, hidden_dim, dropout=0.1):
        super().__init__()
        self.row_enc = nn.Linear(n_users, hidden_dim, bias=False)
        self.col_enc = nn.Linear(n_users, hidden_dim, bias=False)
        self.strength_enc = nn.Linear(1, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, adj_delta):
        if adj_delta is None:
            return None

        adj_delta = ensure_tensor(adj_delta)

        if adj_delta.is_sparse:
            adj_delta = adj_delta.to_dense()

        if adj_delta.dim() == 2:
            adj_delta = adj_delta.unsqueeze(0)

        if adj_delta.dim() != 3:
            raise ValueError(f"adj_delta must be [B,N,N] or [N,N], got {adj_delta.shape}")

        row_feat = self.act(self.row_enc(adj_delta))
        col_feat = self.act(self.col_enc(adj_delta.transpose(-1, -2)))

        strength = adj_delta.abs().sum(dim=-1, keepdim=True)
        strength_feat = self.act(self.strength_enc(strength))

        return self.ln(self.dropout(row_feat + col_feat + strength_feat))


# ============================================================
# Fusion Gate
# ============================================================
class FusionGate(nn.Module):
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.gate = nn.Sequential(
            nn.Linear(dim * n_streams, n_streams),
            nn.Softmax(dim=-1)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, *streams):
        streams = list(streams)
        if len(streams) != self.n_streams:
            raise ValueError(f"FusionGate expected {self.n_streams} streams, got {len(streams)}")

        batch_size = None
        for i in range(len(streams)):
            if streams[i].dim() == 2:
                streams[i] = streams[i].unsqueeze(0)
            batch_size = streams[i].size(0) if batch_size is None else batch_size

        for i in range(len(streams)):
            if streams[i].size(0) == 1 and batch_size > 1:
                streams[i] = streams[i].expand(batch_size, -1, -1)

        concat = torch.cat(streams, dim=-1)
        weights = self.gate(concat)

        fused = 0
        for w, s in zip(weights.unbind(dim=-1), streams):
            fused = fused + w.unsqueeze(-1) * s

        residual = self.res_scale * (streams[0] + streams[1])
        return fused + residual


# ============================================================
# Dense GCN
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(output_dim)

    def _add_self_loop_and_norm(self, adj):
        if adj.dim() == 2:
            n = adj.size(0)
            eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
            adj = adj + eye
            adj = dense_row_normalize(adj)
        elif adj.dim() == 3:
            b, n, _ = adj.shape
            eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(b, -1, -1)
            adj = adj + eye
            adj = dense_row_normalize(adj)
        else:
            raise ValueError(f"adj must be 2D or 3D, got {adj.shape}")
        return adj

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

        if adj.size(0) == 1 and features.size(0) > 1:
            adj = adj.expand(features.size(0), -1, -1)
        if features.size(0) == 1 and adj.size(0) > 1:
            features = features.expand(adj.size(0), -1, -1)

        adj = self._add_self_loop_and_norm(adj)

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
    def __init__(
        self,
        input_dim,
        g_hidden_dim,
        hidden_dim,
        action_dim,
        max_action=1.0,
        temporal_dim=40,
        n_users=None,
        history_len=5
    ):
        super().__init__()

        self.g_hidden_dim = g_hidden_dim
        self.gcn_out = GCNEncoder(input_dim, g_hidden_dim)
        self.gcn_in = GCNEncoder(input_dim, g_hidden_dim)

        self.temporal_enc = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=temporal_dim,
            max_len=history_len
        )

        self.evo_enc = GraphEvolutionEncoder(n_users, g_hidden_dim) if n_users else None
        self.t_align = nn.Linear(temporal_dim, g_hidden_dim)
        self.gate = FusionGate(dim=g_hidden_dim, n_streams=4)

        self.net = nn.Sequential(
            nn.Linear(g_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, features, adj_out, adj_in, history_features=None, adj_delta=None):
        if features.dim() == 2:
            features = features.unsqueeze(0)

        b, n, _ = features.shape

        adj_out = expand_batch(adj_out, b)
        adj_in = expand_batch(adj_in, b)
        history_features = ensure_tensor(history_features)
        adj_delta = ensure_tensor(adj_delta)

        out_emb = self.gcn_out(features, adj_out)
        in_emb = self.gcn_in(features, adj_in)

        t_emb = torch.zeros(b, n, self.g_hidden_dim, device=features.device)
        if history_features is not None:
            t_raw = self.temporal_enc(history_features)
            if t_raw is not None:
                t_emb = self.t_align(t_raw)
                t_emb = expand_batch(t_emb, b)

        evo_emb = torch.zeros(b, n, self.g_hidden_dim, device=features.device)
        if adj_delta is not None and self.evo_enc is not None:
            evo_raw = self.evo_enc(adj_delta)
            if evo_raw is not None:
                evo_emb = evo_raw
                evo_emb = expand_batch(evo_emb, b)

        fused = self.gate(out_emb, in_emb, t_emb, evo_emb)
        action = self.max_action * self.net(fused)
        return action


# ============================================================
# Critic
# ============================================================
class Critic(nn.Module):
    def __init__(
        self,
        input_dim,
        g_hidden_dim,
        hidden_dim,
        action_dim,
        temporal_dim=40,
        n_users=None,
        history_len=5
    ):
        super().__init__()

        self.g_hidden_dim = g_hidden_dim
        fused_dim = g_hidden_dim + action_dim

        self.gcn_out_1 = GCNEncoder(input_dim, g_hidden_dim)
        self.gcn_in_1 = GCNEncoder(input_dim, g_hidden_dim)
        self.t_enc_1 = TemporalEncoder(input_dim, temporal_dim, max_len=history_len)
        self.t_align_1 = nn.Linear(temporal_dim, g_hidden_dim)
        self.evo_enc_1 = GraphEvolutionEncoder(n_users, g_hidden_dim) if n_users else None
        self.gate_1 = FusionGate(dim=g_hidden_dim, n_streams=4)

        self.q1_net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.q1_pool = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.gcn_out_2 = GCNEncoder(input_dim, g_hidden_dim)
        self.gcn_in_2 = GCNEncoder(input_dim, g_hidden_dim)
        self.t_enc_2 = TemporalEncoder(input_dim, temporal_dim, max_len=history_len)
        self.t_align_2 = nn.Linear(temporal_dim, g_hidden_dim)
        self.evo_enc_2 = GraphEvolutionEncoder(n_users, g_hidden_dim) if n_users else None
        self.gate_2 = FusionGate(dim=g_hidden_dim, n_streams=4)

        self.q2_net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.q2_pool = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _encode(self, features, adj_out, adj_in, action,
                gcn_out, gcn_in, t_enc, t_align, evo_enc, gate,
                history_features, adj_delta):
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if action.dim() == 2:
            action = action.unsqueeze(0)

        b, n, _ = features.shape
        adj_out = expand_batch(adj_out, b)
        adj_in = expand_batch(adj_in, b)

        out_emb = gcn_out(features, adj_out)
        in_emb = gcn_in(features, adj_in)

        t_emb = torch.zeros(b, n, self.g_hidden_dim, device=features.device)
        if history_features is not None:
            t_raw = t_enc(history_features)
            if t_raw is not None:
                t_emb = t_align(t_raw)
                t_emb = expand_batch(t_emb, b)

        evo_emb = torch.zeros(b, n, self.g_hidden_dim, device=features.device)
        if adj_delta is not None and evo_enc is not None:
            evo_raw = evo_enc(adj_delta)
            if evo_raw is not None:
                evo_emb = evo_raw
                evo_emb = expand_batch(evo_emb, b)

        fused = gate(out_emb, in_emb, t_emb, evo_emb)
        sa = torch.cat([fused, action], dim=-1)
        return sa

    def _pool_q(self, node_q, pool_score):
        att = F.softmax(pool_score.squeeze(-1), dim=1).unsqueeze(-1)
        return (node_q * att).sum(dim=1)

    def forward(self, features, adj_out, adj_in, action, history_features=None, adj_delta=None):
        sa1 = self._encode(
            features, adj_out, adj_in, action,
            self.gcn_out_1, self.gcn_in_1,
            self.t_enc_1, self.t_align_1,
            self.evo_enc_1, self.gate_1,
            history_features, adj_delta
        )
        node_q1 = self.q1_net(sa1)
        pool1 = self.q1_pool(sa1)
        q1 = self._pool_q(node_q1, pool1)

        sa2 = self._encode(
            features, adj_out, adj_in, action,
            self.gcn_out_2, self.gcn_in_2,
            self.t_enc_2, self.t_align_2,
            self.evo_enc_2, self.gate_2,
            history_features, adj_delta
        )
        node_q2 = self.q2_net(sa2)
        pool2 = self.q2_pool(sa2)
        q2 = self._pool_q(node_q2, pool2)

        return q1, q2

    def Q1(self, features, adj_out, adj_in, action, history_features=None, adj_delta=None):
        sa1 = self._encode(
            features, adj_out, adj_in, action,
            self.gcn_out_1, self.gcn_in_1,
            self.t_enc_1, self.t_align_1,
            self.evo_enc_1, self.gate_1,
            history_features, adj_delta
        )
        node_q1 = self.q1_net(sa1)
        pool1 = self.q1_pool(sa1)
        return self._pool_q(node_q1, pool1)


# ============================================================
# DyNAI
# ============================================================
class DyNAI(nn.Module):
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
        temporal_dim=40,
        **kwargs
    ):
        super().__init__()

        self.per_user_dim = int(state_dim / num_of_users)
        self.num_of_users = num_of_users
        self.history_len = history_len

        self.actor = Actor(
            input_dim=self.per_user_dim,
            g_hidden_dim=g_hidden_dim,
            hidden_dim=fc_hidden_dim,
            action_dim=action_dim,
            max_action=max_action,
            temporal_dim=temporal_dim,
            n_users=num_of_users,
            history_len=history_len
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

        self.critic = Critic(
            input_dim=self.per_user_dim,
            g_hidden_dim=g_hidden_dim,
            hidden_dim=fc_hidden_dim,
            action_dim=action_dim,
            temporal_dim=temporal_dim,
            n_users=num_of_users,
            history_len=history_len
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
        self.adjs_out = None
        self.adjs_in = None
        self.last_actor_loss = float("nan")

    def set_adjs(self, adjs_out, adjs_in):
        self.adjs_out = dense_row_normalize(to_dense_adj(adjs_out))
        self.adjs_in = dense_row_normalize(to_dense_adj(adjs_in))

    def select_action(self, state, history_features=None, adj_delta=None):
        state = ensure_tensor(state)

        if state.dim() == 1:
            state = state.view(1, self.num_of_users, self.per_user_dim)
        elif state.dim() == 2:
            state = state.unsqueeze(0)
        elif state.dim() != 3:
            raise ValueError(f"state must be 1D/2D/3D, got {state.shape}")

        history_features = ensure_tensor(history_features)
        if history_features is not None:
            if history_features.dim() == 3:
                history_features = history_features.unsqueeze(0)
            elif history_features.dim() != 4:
                raise ValueError(
                    f"history_features must be [S,N,D] or [B,S,N,D], got {history_features.shape}"
                )

        adj_delta = ensure_tensor(adj_delta)
        if adj_delta is not None:
            if adj_delta.dim() == 2:
                adj_delta = adj_delta.unsqueeze(0)
            elif adj_delta.dim() != 3:
                raise ValueError(
                    f"adj_delta must be [N,N] or [B,N,N], got {adj_delta.shape}"
                )

        adjs_out = self.adjs_out
        adjs_in = self.adjs_in
        if adjs_out is None or adjs_in is None:
            raise ValueError("adjs_out / adjs_in 还没有通过 set_adjs() 设置")

        adjs_out = ensure_tensor(adjs_out)
        adjs_in = ensure_tensor(adjs_in)

        if adjs_out.dim() == 2:
            adjs_out = adjs_out.unsqueeze(0)
        if adjs_in.dim() == 2:
            adjs_in = adjs_in.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(
                state, adjs_out, adjs_in,
                history_features=history_features,
                adj_delta=adj_delta
            )

        return action.squeeze(0).flatten()

    def soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self, replay_buffer, batch_size):
        self.total_it += 1

        batch = replay_buffer.sample(batch_size, with_aux=True)
        if len(batch) < 11:
            raise ValueError(
                f"ReplayBuffer.sample(with_aux=True) should return at least 11 items, got {len(batch)}"
            )

        (states, actions, rewards, next_states, dones,
         adjs_out, adjs_in, next_adjs_out, next_adjs_in,
         history_features, adj_deltas) = batch[:11]

        states = states.view(batch_size, self.num_of_users, self.per_user_dim)
        next_states = next_states.view(batch_size, self.num_of_users, self.per_user_dim)
        actions = actions.view(batch_size, self.num_of_users, -1)
        rewards = rewards.view(batch_size, 1)
        dones = dones.view(batch_size, 1)

        if actions.size(-1) != 1:
            actions = actions[..., :1]

        adjs_out = dense_row_normalize(adjs_out)
        adjs_in = dense_row_normalize(adjs_in)
        next_adjs_out = dense_row_normalize(next_adjs_out)
        next_adjs_in = dense_row_normalize(next_adjs_in)

        history_features = ensure_tensor(history_features)
        adj_deltas = ensure_tensor(adj_deltas)

        if history_features is not None:
            if history_features.dim() == 3:
                history_features = history_features.unsqueeze(0)
            elif history_features.dim() != 4:
                raise ValueError(
                    f"history_features must be [B,S,N,D] or [S,N,D], got {history_features.shape}"
                )

        if adj_deltas is not None:
            if adj_deltas.dim() == 2:
                adj_deltas = adj_deltas.unsqueeze(0)
            elif adj_deltas.dim() != 3:
                raise ValueError(
                    f"adj_deltas must be [B,N,N] or [N,N], got {adj_deltas.shape}"
                )

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(
                next_states, next_adjs_out, next_adjs_in,
                history_features=history_features,
                adj_delta=adj_deltas
            )
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(
                next_states, next_adjs_out, next_adjs_in,
                next_action,
                history_features=history_features,
                adj_delta=adj_deltas
            )
            target_Q = rewards + (1.0 - dones) * self.discount * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(
            states, adjs_out, adjs_in, actions,
            history_features=history_features,
            adj_delta=adj_deltas
        )

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_update = False
        if self.total_it % self.policy_freq == 0:
            pred_action = self.actor(
                states, adjs_out, adjs_in,
                history_features=history_features,
                adj_delta=adj_deltas
            )
            actor_loss = -self.critic.Q1(
                states, adjs_out, adjs_in, pred_action,
                history_features=history_features,
                adj_delta=adj_deltas
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.soft_update()
            self.last_actor_loss = actor_loss.item()
            actor_update = True

        return critic_loss.item(), self.last_actor_loss, actor_update

