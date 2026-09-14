import os
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import networkx as nx
import pandas as pd


item_num = 4

# 固定随机种子，保证可复现
np.random.seed(42)

# ============================================================
# 项目根目录与统一路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "Results"
LOG_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def read_dataset(
    filename,
    snapshot_num=20,
    top_k=3,
    time_start=None,
    time_end=None,
    max_events=None,
    txt_format="time_source_target",
    delimiter=None
):
    """
    读取原始动态图数据，并统一转换为：
    (source, target, time)

    适配格式：
    - 无表头 TSV: time \t source \t target
    - 无表头 TXT: time source target
    - CSV/TSV 有表头情况：可兼容常见列名

    输出：
    1) Dataset/{filename}/{filename}_preference.csv
    2) Dataset/{filename}/{filename}_network.csv
    """
    base_dir = DATASET_DIR / filename

    # 按优先级查找文件
    for ext in [".txt", ".csv", ".tsv"]:
        dataset_path = base_dir / f"{filename}{ext}"
        if dataset_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"原始数据文件不存在: 尝试了 .txt/.csv/.tsv 在目录 {base_dir}"
        )

    # ------------------------------------------------------------
    # 自动识别文件分隔符
    # ------------------------------------------------------------
    reddit_mode = False
    if delimiter is None:
        if dataset_path.suffix == ".tsv":
            delimiter = "\t"
        elif dataset_path.suffix == ".csv":
            delimiter = ","
        else:
            delimiter = None

    network = {"Source": [], "Target": [], "Weight": [], "timestamp": []}
    users_preference = {}
    raw_data = []

    # reddit 风格映射（保留原逻辑）
    if reddit_mode:
        node_to_id = {}
        next_id = 0

        def get_node_id(name):
            nonlocal next_id
            if name not in node_to_id:
                node_to_id[name] = next_id
                next_id += 1
            return node_to_id[name]

    def add_preference_if_needed(node_id):
        if node_id not in users_preference:
            users_preference[node_id] = [node_id] + [np.random.uniform(0, 1) for _ in range(item_num)]

    def pass_time_filter(time):
        if time_start is not None and time < time_start:
            return False
        if time_end is not None and time > time_end:
            return False
        return True

    # ------------------------------------------------------------
    # 读取数据
    # ------------------------------------------------------------
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        if delimiter is None:
            # .txt：空白分隔，无表头
            for line in f:
                data = line.strip().split()
                if len(data) < 3:
                    continue

                try:
                    if txt_format == "time_source_target":
                        time = int(float(data[0]))
                        source = int(data[1])
                        target = int(data[2])
                    elif txt_format == "source_target_time":
                        source = int(data[0])
                        target = int(data[1])
                        time = int(float(data[2]))
                    elif txt_format == "source_time_target":
                        source = int(data[0])
                        time = int(float(data[1]))
                        target = int(data[2])
                    else:
                        raise ValueError(
                            f"不支持的 txt_format: {txt_format}. "
                            f"可选: time_source_target, source_target_time, source_time_target"
                        )
                except:
                    continue

                if not pass_time_filter(time):
                    continue

                raw_data.append((source, target, time))
                add_preference_if_needed(source)
                add_preference_if_needed(target)

                if max_events is not None and len(raw_data) >= max_events:
                    break
        else:
            # --------------------------------------------------------
            # TSV/CSV 统一读取
            # 重点：你的 TSV 是无表头，所以 header=None
            # --------------------------------------------------------
            if dataset_path.suffix == ".tsv":
                df = pd.read_csv(dataset_path, sep="\t", header=None, usecols=[0, 1, 2])
            else:
                df = pd.read_csv(dataset_path, sep=delimiter)

            # 如果是有表头文件，这里统一转小写；无表头则保持位置访问
            if df.columns.dtype == "int64" or isinstance(df.columns[0], (int, np.integer)):
                # 无表头：直接按位置解析前三列
                pass
            else:
                # 有表头：去掉可能存在的 Unnamed 列，并标准化列名
                if len(df.columns) > 0 and (str(df.columns[0]).startswith("Unnamed") or df.columns[0] == ""):
                    df = df.iloc[:, 1:]
                df.columns = [str(c).strip().lower() for c in df.columns]

            # --------------------------------------------------------
            # 无表头 TSV：直接按前三列读取
            # --------------------------------------------------------
            if dataset_path.suffix == ".tsv" and (df.columns.dtype == "int64" or isinstance(df.columns[0], (int, np.integer))):
                for _, row in df.iterrows():
                    try:
                        time = int(float(row.iloc[0]))
                        source = int(row.iloc[1])
                        target = int(row.iloc[2])
                    except:
                        continue

                    if not pass_time_filter(time):
                        continue

                    raw_data.append((source, target, time))
                    add_preference_if_needed(source)
                    add_preference_if_needed(target)

                    if max_events is not None and len(raw_data) >= max_events:
                        break

            # --------------------------------------------------------
            # reddit 风格 TSV（保留兼容）
            # --------------------------------------------------------
            elif reddit_mode:
                for _, row in df.iterrows():
                    if len(row) < 4:
                        continue

                    source_name = str(row.iloc[0]).strip()
                    target_name = str(row.iloc[1]).strip()
                    time_str = str(row.iloc[3]).strip()

                    try:
                        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        time = int(dt.timestamp())
                    except:
                        continue

                    if not pass_time_filter(time):
                        continue

                    source = get_node_id(source_name)
                    target = get_node_id(target_name)

                    raw_data.append((source, target, time))
                    add_preference_if_needed(source)
                    add_preference_if_needed(target)

                    if max_events is not None and len(raw_data) >= max_events:
                        break

            # --------------------------------------------------------
            # 有表头的常见列名情况
            # --------------------------------------------------------
            else:
                cols = set(df.columns)

                # 情况 A：time, source, target
                if {"time", "source", "target"}.issubset(cols):
                    for _, row in df.iterrows():
                        try:
                            time = int(float(row["time"]))
                            source = int(row["source"])
                            target = int(row["target"])
                        except:
                            continue

                        if not pass_time_filter(time):
                            continue

                        raw_data.append((source, target, time))
                        add_preference_if_needed(source)
                        add_preference_if_needed(target)

                        if max_events is not None and len(raw_data) >= max_events:
                            break

                # 情况 B：timestamp, source, target
                elif {"timestamp", "source", "target"}.issubset(cols):
                    for _, row in df.iterrows():
                        try:
                            time = int(float(row["timestamp"]))
                            source = int(row["source"])
                            target = int(row["target"])
                        except:
                            continue

                        if not pass_time_filter(time):
                            continue

                        raw_data.append((source, target, time))
                        add_preference_if_needed(source)
                        add_preference_if_needed(target)

                        if max_events is not None and len(raw_data) >= max_events:
                            break

                # 情况 C：source, target, time
                elif {"source", "target", "time"}.issubset(cols):
                    for _, row in df.iterrows():
                        try:
                            source = int(row["source"])
                            target = int(row["target"])
                            time = int(float(row["time"]))
                        except:
                            continue

                        if not pass_time_filter(time):
                            continue

                        raw_data.append((source, target, time))
                        add_preference_if_needed(source)
                        add_preference_if_needed(target)

                        if max_events is not None and len(raw_data) >= max_events:
                            break

                # 情况 D：contact_time, day, id1, id2
                elif {"contact_time", "day", "id1", "id2"}.issubset(cols):
                    for _, row in df.iterrows():
                        try:
                            source = int(row["id1"])
                            target = int(row["id2"])
                            time = int(row["day"]) * 100000 + int(row["contact_time"])
                        except:
                            continue

                        if not pass_time_filter(time):
                            continue

                        raw_data.append((source, target, time))
                        add_preference_if_needed(source)
                        add_preference_if_needed(target)

                        if max_events is not None and len(raw_data) >= max_events:
                            break

                # 情况 E：至少 3 列，默认按前三列 time, source, target 解析
                elif len(df.columns) >= 3:
                    for _, row in df.iterrows():
                        try:
                            time = int(float(row.iloc[0]))
                            source = int(row.iloc[1])
                            target = int(row.iloc[2])
                        except:
                            continue

                        if not pass_time_filter(time):
                            continue

                        raw_data.append((source, target, time))
                        add_preference_if_needed(source)
                        add_preference_if_needed(target)

                        if max_events is not None and len(raw_data) >= max_events:
                            break
                else:
                    print("当前文件列名为：", list(df.columns))
                    raise ValueError(f"无法识别 CSV/TSV 格式: {dataset_path}")

    if len(raw_data) == 0:
        raise ValueError(f"Dataset is empty after filtering: {dataset_path}")

    # ============================================================
    # 按原逻辑切 snapshot
    # ============================================================
    time_list = [x[2] for x in raw_data]
    min_time = min(time_list)
    max_time = max(time_list)

    snapshot_num = min(snapshot_num, 20)
    total_range = max_time - min_time + 1
    window_size = int(np.ceil(total_range / snapshot_num))
    stride = max(1, window_size // 2)

    windows = []
    start = min_time
    while start <= max_time:
        end = start + window_size
        windows.append((start, end))
        start += stride

    windows = windows[:snapshot_num]

    for t_idx, (start, end) in enumerate(windows):
        edge_count = {}

        for source, target, time in raw_data:
            if start <= time < end:
                key = (source, target)
                edge_count[key] = edge_count.get(key, 0) + 1

        target_edges = {}
        for (source, target), count in edge_count.items():
            if target not in target_edges:
                target_edges[target] = []
            target_edges[target].append((source, count))

        for target, edges in target_edges.items():
            edges = sorted(edges, key=lambda x: x[1], reverse=True)
            edges = edges[:top_k]

            total = sum(count for _, count in edges)
            if total == 0:
                continue

            for source, count in edges:
                weight = count / float(total)
                weight = min(1.0, weight)

                network["Source"].append(source)
                network["Target"].append(target)
                network["Weight"].append(weight)
                network["timestamp"].append(t_idx)

    # ============================================================
    # 保存文件
    # ============================================================
    output_dir = DATASET_DIR / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    preference_path = output_dir / f"{filename}_preference.csv"
    network_path = output_dir / f"{filename}_network.csv"

    df_preference = pd.DataFrame.from_dict(users_preference, orient="index")
    pref_cols = ["ID"] + [str(i) for i in range(1, item_num + 1)]
    df_preference.columns = pref_cols
    df_preference.to_csv(preference_path, index=False)

    df_network = pd.DataFrame.from_dict(network)
    df_network.to_csv(network_path, index=False)

    if reddit_mode:
        mapping_path = output_dir / f"{filename}_node_mapping.csv"
        mapping_df = pd.DataFrame(list(node_to_id.items()), columns=["Subreddit", "ID"])
        mapping_df.to_csv(mapping_path, index=False)
        print(f"节点映射保存至: {mapping_path}")

    print("数据预处理完成")
    print("user num:", len(users_preference))
    print("snapshot num:", len(set(network["timestamp"])))
    print("window size:", window_size)
    print("stride:", stride)
    print("preference saved to:", preference_path)
    print("network saved to:", network_path)


def import_users_data_from_csv(filename):
    preference_path = DATASET_DIR / filename / f"{filename}_preference.csv"
    network_path = DATASET_DIR / filename / f"{filename}_network.csv"

    if not preference_path.exists():
        raise FileNotFoundError(f"用户偏好文件不存在: {preference_path}")

    if not network_path.exists():
        raise FileNotFoundError(f"网络文件不存在: {network_path}")

    network = pd.read_csv(network_path)
    users_preference = pd.read_csv(preference_path)

    graphs = {}
    snapshot_list = sorted(network["timestamp"].unique())

    for t in snapshot_list:
        df_t = network[network["timestamp"] == t]

        G = nx.from_pandas_edgelist(
            df_t,
            source="Source",
            target="Target",
            edge_attr=["Weight"],
            create_using=nx.DiGraph()
        )

        graphs[t] = G

    all_nodes = set()
    for t in graphs:
        all_nodes.update(graphs[t].nodes())

    preference_dict = {}
    ID_dict = {}

    for _, row in users_preference.iterrows():
        uid = int(row.iloc[0])
        pref = [float(x) for x in row.iloc[1:].tolist()]

        preference_dict[uid] = pref
        ID_dict[uid] = uid

    for t in graphs:
        G = graphs[t]

        for node in all_nodes:
            if node not in G:
                G.add_node(node)

        nx.set_node_attributes(G, ID_dict, "ID")
        nx.set_node_attributes(G, preference_dict, "Preference")

    return graphs, sorted(list(all_nodes)), snapshot_list


def save_results(dataset, results, method):
    result_dir = RESULTS_DIR / dataset
    result_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{method}.txt"

    with open(result_path, mode="w", encoding="utf-8") as f:
        for data in results:
            f.write(str(data) + "\n")

    print(f"结果已保存至: {result_path}")


def process_log(dataset):
    train_result_dir = RESULTS_DIR / "train"
    input_path = train_result_dir / f"{dataset}.out"
    output_path = train_result_dir / f"{dataset}.txt"

    if not input_path.exists():
        raise FileNotFoundError(f"训练日志文件不存在: {input_path}")

    results = []

    with open(input_path, mode="r", encoding="utf-8") as f:
        for index, line in enumerate(f.readlines()):
            data = line.strip().split(" ")

            if len(data) < 7:
                continue

            if (index != 0 and data[1] == "0") or data[0] != "Epi:":
                continue

            results.append(data[6])

    with open(output_path, mode="w", encoding="utf-8") as f:
        for data in results:
            f.write(data + "\n")

    print(f"处理后的日志已保存至: {output_path}")


def init_train_log(log_path):
    import csv
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "train_reward",
            "train_avg_active_rate",
            "train_budget_left",
            "critic_loss",
            "actor_loss",
            "actor_update",
            "eval_reward",
            "eval_active_rate",
            "eval_budget_left"
        ])


def append_train_log(file_path, row):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == "__main__":
    read_dataset(
        "ht2009_contact_list",
        snapshot_num=20,
        top_k=3,
        delimiter="\t"
    )