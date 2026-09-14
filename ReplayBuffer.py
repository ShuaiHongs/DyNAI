import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
        self,
        capacity,
        device=None,
        history_len=5,
        num_of_users=None,
        per_user_dim=None,
        return_dense_adj=True
    ):
        self.capacity = int(capacity)
        self.target_device = device if device is not None else torch.device("cpu")
        self.history_len = history_len
        self.num_of_users = num_of_users
        self.per_user_dim = per_user_dim
        self.return_dense_adj = return_dense_adj

        self._prev_adj_out = None
        self.buffer = []
        self.position = 0

        # 存历史状态，长度固定为 history_len
        self.history_queue = deque(maxlen=history_len)

    # ============================================================
    # 基础转换工具
    # ============================================================
    def _to_tensor(self, x, dtype=torch.float32, to_device=False, dense_if_sparse=True):
        if x is None:
            return None

        if torch.is_tensor(x):
            t = x.detach().clone()
            if dense_if_sparse and t.is_sparse:
                t = t.to_dense()
            t = t.to(torch.device("cpu")).to(dtype=dtype)
        elif isinstance(x, np.ndarray):
            t = torch.tensor(x, dtype=dtype, device=torch.device("cpu"))
        elif isinstance(x, (list, tuple)):
            t = torch.tensor(np.asarray(x), dtype=dtype, device=torch.device("cpu"))
        else:
            t = torch.tensor(float(x), dtype=dtype, device=torch.device("cpu"))

        if to_device and self.target_device.type != "cpu":
            t = t.to(self.target_device)

        return t

    def _check_shape_info(self):
        if self.num_of_users is None or self.per_user_dim is None:
            raise ValueError("ReplayBuffer 需要传入 num_of_users 和 per_user_dim。")

    def _ensure_state_shape(self, state_t):
        """
        统一 state 形状为 [num_of_users, per_user_dim]
        """
        self._check_shape_info()

        if state_t is None:
            raise ValueError("state 不能为空")

        if state_t.dim() == 1:
            if state_t.numel() != self.num_of_users * self.per_user_dim:
                raise ValueError(
                    f"state size mismatch: got {state_t.numel()}, "
                    f"expected {self.num_of_users * self.per_user_dim}"
                )
            state_t = state_t.view(self.num_of_users, self.per_user_dim)

        elif state_t.dim() == 2:
            if state_t.shape != (self.num_of_users, self.per_user_dim):
                raise ValueError(
                    f"state shape mismatch: got {state_t.shape}, "
                    f"expected ({self.num_of_users}, {self.per_user_dim})"
                )
        else:
            raise ValueError(f"state must be 1D or 2D, got shape {state_t.shape}")

        return state_t

    def _ensure_adj_shape(self, adj_t):
        """
        接受 [N,N] 或 [B,N,N]，不强行改形状，只做基础检查
        """
        if adj_t is None:
            raise ValueError("adj 不能为空")

        if adj_t.dim() not in (2, 3):
            raise ValueError(f"adj must be 2D or 3D, got {adj_t.shape}")

        return adj_t

    # ============================================================
    # history features
    # ============================================================
    def _build_history_tensor(self, to_device=False):
        self._check_shape_info()

        # 如果历史不足，用零补齐
        if len(self.history_queue) == 0:
            t = torch.zeros(
                self.history_len,
                self.num_of_users,
                self.per_user_dim,
                device=torch.device("cpu"),
                dtype=torch.float32
            )
        else:
            history_list = list(self.history_queue)

            if len(history_list) < self.history_len:
                pad_len = self.history_len - len(history_list)
                zero_pad = [
                    torch.zeros(
                        self.num_of_users,
                        self.per_user_dim,
                        dtype=torch.float32,
                        device=torch.device("cpu")
                    )
                    for _ in range(pad_len)
                ]
                history_list = zero_pad + history_list

            frames = []
            for h in history_list:
                h = self._to_tensor(h, dtype=torch.float32, to_device=False)
                h = self._ensure_state_shape(h)
                frames.append(h)

            t = torch.stack(frames, dim=0)  # [S, N, D]

        if to_device and self.target_device.type != "cpu":
            t = t.to(self.target_device)

        return t

    def get_history_features(self):
        """
        返回 [1, S, N, D]
        """
        return self._build_history_tensor(to_device=True).unsqueeze(0)

    # ============================================================
    # adjacency delta
    # ============================================================
    def get_adj_delta(self, curr_adj_out):
        """
        在环境外部想手动拿 adjacency 差分，可以用这个。
        返回形状与 curr_adj_out 一致。
        """
        curr_adj_cpu = self._to_tensor(curr_adj_out, dtype=torch.float32, to_device=False)

        if self._prev_adj_out is None:
            delta_cpu = torch.zeros_like(curr_adj_cpu)
        else:
            if curr_adj_cpu.shape != self._prev_adj_out.shape:
                raise ValueError(
                    f"adj shape changed: current {curr_adj_cpu.shape}, "
                    f"prev {self._prev_adj_out.shape}"
                )
            delta_cpu = torch.abs(curr_adj_cpu - self._prev_adj_out)

        self._prev_adj_out = curr_adj_cpu.clone()

        if self.target_device.type != "cpu":
            delta_cpu = delta_cpu.to(self.target_device)

        return delta_cpu

    # ============================================================
    # push transition
    # ============================================================
    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        adj_out,
        adj_in,
        next_adj_out,
        next_adj_in
    ):
        self._check_shape_info()

        # state / next_state
        state_t = self._to_tensor(state, dtype=torch.float32, to_device=False)
        state_t = self._ensure_state_shape(state_t)

        if next_state is None:
            next_state_t = state_t.clone()
        else:
            next_state_t = self._to_tensor(next_state, dtype=torch.float32, to_device=False)
            next_state_t = self._ensure_state_shape(next_state_t)

        # action / reward / done
        action_t = self._to_tensor(action, dtype=torch.float32, to_device=False).flatten()
        reward_t = self._to_tensor(reward, dtype=torch.float32, to_device=False).view(1)
        done_t = self._to_tensor(float(done), dtype=torch.float32, to_device=False).view(1)

        # adjacency
        adj_out_t = self._to_tensor(adj_out, dtype=torch.float32, to_device=False)
        adj_out_t = self._ensure_adj_shape(adj_out_t)

        adj_in_t = self._to_tensor(adj_in, dtype=torch.float32, to_device=False)
        adj_in_t = self._ensure_adj_shape(adj_in_t)

        next_adj_out_t = self._to_tensor(next_adj_out, dtype=torch.float32, to_device=False)
        next_adj_out_t = self._ensure_adj_shape(next_adj_out_t)

        next_adj_in_t = self._to_tensor(next_adj_in, dtype=torch.float32, to_device=False)
        next_adj_in_t = self._ensure_adj_shape(next_adj_in_t)

        if adj_out_t.shape != next_adj_out_t.shape:
            raise ValueError(
                f"adj_out and next_adj_out shape mismatch: {adj_out_t.shape} vs {next_adj_out_t.shape}"
            )
        if adj_in_t.shape != next_adj_in_t.shape:
            raise ValueError(
                f"adj_in and next_adj_in shape mismatch: {adj_in_t.shape} vs {next_adj_in_t.shape}"
            )

        # history features: 当前 transition 记录的是“动作发生前”的历史
        history_features_t = self._build_history_tensor(to_device=False)

        # adj delta: 用 out-adj 的变化作为图演化信号
        adj_delta_t = torch.abs(next_adj_out_t - adj_out_t)

        data = (
            state_t,
            action_t,
            reward_t,
            next_state_t,
            done_t,
            adj_out_t,
            adj_in_t,
            next_adj_out_t,
            next_adj_in_t,
            history_features_t,
            adj_delta_t
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data

        self.position = (self.position + 1) % self.capacity

        # 把 next_state 放入历史队列，供下一步使用
        self.history_queue.append(next_state_t.clone())

    # ============================================================
    # sample
    # ============================================================
    def sample(self, batch_size, with_aux=False):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")

        batch = random.sample(self.buffer, batch_size)

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            adjs_out,
            adjs_in,
            next_adjs_out,
            next_adjs_in,
            history_features,
            adj_deltas
        ) = zip(*batch)

        states = torch.stack(states).to(self.target_device)
        actions = torch.stack(actions).to(self.target_device)
        rewards = torch.stack(rewards).to(self.target_device).view(-1)
        next_states = torch.stack(next_states).to(self.target_device)
        dones = torch.stack(dones).to(self.target_device).view(-1)

        adjs_out = torch.stack(adjs_out).to(self.target_device)
        adjs_in = torch.stack(adjs_in).to(self.target_device)
        next_adjs_out = torch.stack(next_adjs_out).to(self.target_device)
        next_adjs_in = torch.stack(next_adjs_in).to(self.target_device)

        if not with_aux:
            return states, actions, rewards, next_states, dones

        history_features = torch.stack(history_features).to(self.target_device)
        adj_deltas = torch.stack(adj_deltas).to(self.target_device)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            adjs_out,
            adjs_in,
            next_adjs_out,
            next_adjs_in,
            history_features,
            adj_deltas
        )

    # ============================================================
    # utils
    # ============================================================
    def clear(self):
        self.buffer = []
        self.position = 0
        self.history_queue.clear()
        self._prev_adj_out = None

    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size