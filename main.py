import os
import argparse
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch

from environment import Simulation
from ReplayBuffer import ReplayBuffer
from utils import init_train_log, append_train_log

from GAC import GAC
from DyNAI import DyNAI


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "checkpoints"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fmt(v, ndigits=4):
    """
    将数值格式化为字符串；None / NaN 显示为 -
    """
    if v is None:
        return "-"
    try:
        if np.isnan(v):
            return "-"
    except Exception:
        pass
    return f"{float(v):.{ndigits}f}"


def create_env(dataset_name, num_of_actions, budget):
    return Simulation(dataset_name, num_of_actions, budget)


def build_model(model_name, state_dim, num_of_users, args):
    """
    兼容构造：
    - GAC
    - DyNAI

    注意：如果某个模型不需要某些参数，也通过 **kwargs 兼容掉。
    """
    model_name = model_name.lower()

    common_kwargs = dict(
        state_dim=state_dim,
        g_hidden_dim=args.g_hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        action_dim=1,
        max_action=args.max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        num_of_users=num_of_users,
        history_len=args.history_len,
        temporal_dim=args.temporal_dim,
        num_layer=args.num_layer,
        dropout=args.dropout,
    )

    if model_name == "gac":
        model = GAC(**common_kwargs)
    elif model_name == "dynai":
        model = DyNAI(**common_kwargs)
    else:
        raise ValueError(f"未知模型类型: {model_name}, 只能选择 gac 或 dynai")

    return model.to(device)


def _call_select_action(model, state, history_features=None, adj_delta=None):
    """
    统一调用 select_action，兼容：
    - select_action(state)
    - select_action(state, history_features=..., adj_delta=...)
    """
    try:
        return model.select_action(
            state,
            history_features=history_features,
            adj_delta=adj_delta
        )
    except TypeError:
        return model.select_action(state)


def select_action_by_model(model, model_name, state, replay_buffer, curr_adj_out):
    """
    根据模型类型选择 action。
    dynai 会额外提供 history_features 和 adj_delta。
    """
    model_name = model_name.lower()

    with torch.no_grad():
        if model_name == "dynai":
            history_features = replay_buffer.get_history_features()
            adj_delta = replay_buffer.get_adj_delta(curr_adj_out)
            action = _call_select_action(
                model=model,
                state=state,
                history_features=history_features,
                adj_delta=adj_delta
            )
        else:
            action = _call_select_action(model=model, state=state)

    if torch.is_tensor(action):
        action = action.detach().cpu().numpy().astype(np.float32)

    return action


def _extract_reset_out(reset_out, env):
    """
    兼容 reset 返回值长度：
    - (state, rate, active_num)
    - (state, rate, active_num, remaining_budget)
    """
    if len(reset_out) == 4:
        state, rate, active_num, remaining_budget = reset_out
    else:
        state, rate, active_num = reset_out
        remaining_budget = env.remaining_budget.cpu().numpy()
    return state, rate, active_num, remaining_budget


def _call_train_step(model, replay_buffer, batch_size):
    """
    统一调用训练函数，兼容：
    - train_step(replay_buffer, batch_size)
    - train(replay_buffer, batch_size)
    """
    if hasattr(model, "train_step"):
        return model.train_step(replay_buffer, batch_size)
    elif hasattr(model, "train"):
        return model.train(replay_buffer, batch_size)
    else:
        raise AttributeError("模型没有 train_step 或 train 方法")


def evaluate_policy(model, model_name, env, episodes=3, history_len=5, per_user_dim=None):
    rewards = []
    active_rates = []
    budget_lefts = []

    for _ in range(episodes):
        reset_out = env.reset()
        state, _, _, remaining_budget = _extract_reset_out(reset_out, env)

        model.set_adjs(env.adjacency_matrix_out, env.adjacency_matrix_in)

        eval_buffer = ReplayBuffer(
            capacity=1000,
            device=device,
            history_len=history_len,
            num_of_users=env.num_of_user,
            per_user_dim=per_user_dim,
            return_dense_adj=True
        )

        ep_reward = 0.0
        ep_active = []
        done = False
        steps = 0

        while not done and steps < len(env.snapshot_list):
            curr_adj_out = env.adjacency_matrix_out

            action = select_action_by_model(
                model=model,
                model_name=model_name,
                state=state,
                replay_buffer=eval_buffer,
                curr_adj_out=curr_adj_out
            )

            next_state, reward, rate, _, remaining_budget, done = env.step(action)

            # 只维护 history_queue，一次即可，不要重复塞
            eval_buffer.history_queue.append(
                torch.tensor(state, dtype=torch.float32, device=torch.device("cpu"))
            )

            ep_reward += float(reward)
            ep_active.append(float(rate))
            state = next_state

            if not done:
                model.set_adjs(env.adjacency_matrix_out, env.adjacency_matrix_in)

            steps += 1

        rewards.append(ep_reward)
        active_rates.append(np.mean(ep_active) if len(ep_active) > 0 else 0.0)
        budget_lefts.append(
            float(remaining_budget[0]) if np.ndim(remaining_budget) > 0 else float(remaining_budget)
        )

    eval_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    eval_active = float(np.mean(active_rates)) if len(active_rates) > 0 else 0.0
    eval_budget_left = float(np.mean(budget_lefts)) if len(budget_lefts) > 0 else 0.0

    return eval_reward, eval_active, eval_budget_left


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    env = create_env(args.dataset, args.num_of_actions, args.budget)

    reset_out = env.reset()
    state, _, _, remaining_budget = _extract_reset_out(reset_out, env)

    state_dim = int(state.shape[0])
    num_of_users = int(env.num_of_user)
    per_user_dim = state_dim // num_of_users

    model = build_model(
        model_name=args.model,
        state_dim=state_dim,
        num_of_users=num_of_users,
        args=args
    )

    model.set_adjs(env.adjacency_matrix_out, env.adjacency_matrix_in)

    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        device=device,
        history_len=args.history_len,
        num_of_users=num_of_users,
        per_user_dim=per_user_dim,
        return_dense_adj=True
    )

    log_path = LOG_DIR / f"{args.dataset}_seed{args.seed}_{args.model}.csv"
    init_train_log(str(log_path))

    total_steps = 0

    print("=" * 80)
    print("[TRAIN START]")
    print(f"Dataset     : {args.dataset}")
    print(f"Model       : {args.model}")
    print(f"Budget      : {args.budget}")
    print(f"Device      : {device}")
    print(f"Log path    : {log_path}")
    print("=" * 80)

    for episode in range(args.max_episodes):
        reset_out = env.reset()
        state, _, _, remaining_budget = _extract_reset_out(reset_out, env)

        model.set_adjs(env.adjacency_matrix_out, env.adjacency_matrix_in)
        replay_buffer._prev_adj_out = None

        ep_reward = 0.0
        ep_active_rates = []
        done = False
        step = 0

        critic_loss = np.nan
        actor_loss = np.nan
        actor_update = False

        while not done and step < len(env.snapshot_list):
            total_steps += 1

            curr_adj_out = env.adjacency_matrix_out
            curr_adj_in = env.adjacency_matrix_in

            if total_steps < args.start_timesteps:
                action = np.random.uniform(
                    -args.max_action,
                    args.max_action,
                    size=(num_of_users,)
                ).astype(np.float32)
            else:
                action = select_action_by_model(
                    model=model,
                    model_name=args.model,
                    state=state,
                    replay_buffer=replay_buffer,
                    curr_adj_out=curr_adj_out
                )

                action = np.asarray(action, dtype=np.float32)

                action += np.random.normal(
                    0,
                    args.policy_noise,
                    size=action.shape
                ).astype(np.float32)

                action = np.clip(action, -args.max_action, args.max_action)

            next_state, reward, rate, _, remaining_budget, done = env.step(action)

            next_adj_out = env.adjacency_matrix_out
            next_adj_in = env.adjacency_matrix_in

            replay_buffer.push(
                state, action, reward, next_state, done,
                curr_adj_out, curr_adj_in,
                next_adj_out, next_adj_in
            )

            # 只维护 history_queue，一次即可，不要重复塞
            replay_buffer.history_queue.append(
                torch.tensor(state, dtype=torch.float32, device=torch.device("cpu"))
            )

            state = next_state
            ep_reward += float(reward)
            ep_active_rates.append(float(rate))

            if replay_buffer.is_ready(args.batch_size) and total_steps >= args.start_timesteps:
                train_out = _call_train_step(model, replay_buffer, args.batch_size)

                if isinstance(train_out, (tuple, list)):
                    critic_loss = train_out[0] if len(train_out) > 0 else np.nan
                    actor_loss = train_out[1] if len(train_out) > 1 else np.nan
                    actor_update = train_out[2] if len(train_out) > 2 else False
                else:
                    critic_loss = train_out
                    actor_loss = np.nan
                    actor_update = False

            if not done:
                model.set_adjs(env.adjacency_matrix_out, env.adjacency_matrix_in)

            step += 1

        avg_active = np.mean(ep_active_rates) if len(ep_active_rates) > 0 else 0.0
        budget_left = float(remaining_budget[0]) if np.ndim(remaining_budget) > 0 else float(remaining_budget)

        eval_reward = None
        eval_active = None
        eval_budget_left = None

        if (episode + 1) % args.eval_interval == 0:
            eval_reward, eval_active, eval_budget_left = evaluate_policy(
                model=model,
                model_name=args.model,
                env=env,
                episodes=args.eval_episodes,
                history_len=args.history_len,
                per_user_dim=per_user_dim
            )

        print(
            f"[{args.model.upper()}] "
            f"Episode {episode + 1:04d} | "
            f"Train Reward = {ep_reward:.4f} | "
            f"Train Active = {avg_active:.4f} | "
            f"Budget Left = {budget_left:.2f} | "
            f"Critic Loss = {fmt(critic_loss, 6)} | "
            f"Actor Loss = {fmt(actor_loss, 6)} | "
            f"Actor Update = {actor_update} | "
            f"Eval Reward = {fmt(eval_reward, 4)} | "
            f"Eval Active = {fmt(eval_active, 4)} | "
            f"Eval Budget Left = {fmt(eval_budget_left, 2)}"
        )

        append_train_log(str(log_path), [
            episode + 1,
            ep_reward,
            avg_active,
            budget_left,
            critic_loss,
            actor_loss,
            int(actor_update),
            eval_reward if eval_reward is not None else np.nan,
            eval_active if eval_active is not None else np.nan,
            eval_budget_left if eval_budget_left is not None else np.nan
        ])

    final_model_path = MODEL_DIR / f"{args.dataset}_seed{args.seed}_{args.model}_final.pth"
    torch.save(model.state_dict(), final_model_path)

    print("=" * 80)
    print("[TRAIN FINISHED]")
    print(f"Model saved to: {final_model_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="email-Eu-core-temporal-Dept1")
    parser.add_argument("--model", type=str, default="dynai", choices=["gac", "dynai"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_of_actions", type=int, default=4)
    parser.add_argument("--budget", type=float, default=30)

    parser.add_argument("--max_episodes", type=int, default=150)
    parser.add_argument("--start_timesteps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=3)

    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--g_hidden_dim", type=int, default=64)
    parser.add_argument("--fc_hidden_dim", type=int, default=128)

    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--temporal_dim", type=int, default=40)

    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--policy_noise", type=float, default=0.05)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)

    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)