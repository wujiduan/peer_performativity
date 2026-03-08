import copy
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pokec_simulations import (
    add_graph_features,
    categorical_features,
    compute_score,
    extract_features,
    feature_labels,
    numerical_features,
    parse_args,
    preprocess,
    run_simulation,
    textual_features,
)


def load_profiles_and_graph(target_column: str):
    profiles_path = f"pokec_dataset/lcc_profiles_{target_column}.pk"
    graph_path = f"pokec_dataset/lcc_graph_{target_column}.pk"

    if os.path.exists(profiles_path) and os.path.exists(graph_path):
        with open(profiles_path, "rb") as f:
            df = pickle.load(f)
        with open(graph_path, "rb") as f:
            network_lcc = pickle.load(f)
        print("lcc user num:", len(df))
        return df, network_lcc

    df = pd.read_csv("pokec_dataset/profiles.txt", sep="\t", header=None)
    feature_len = len(feature_labels)
    df = df.iloc[:, :feature_len]
    df.columns = feature_labels
    df = df.replace(r"^\s*$", np.nan, regex=True)
    mask = df[target_column].isna() | (df[target_column].astype(str).str.strip() == "")
    df = df[~mask].copy()
    df = df.sample(n=30000, random_state=2026)

    df, network_lcc = preprocess(df, target_column, edge_path="pokec_dataset/relationships.txt")
    print("number of nodes in lcc:", len(network_lcc))

    with open(profiles_path, "wb") as f:
        pickle.dump(df, f)

    return df, network_lcc


def load_or_compute_scores(df_labeled, df_unlabeled, target_column, args, total_n):
    param_folder = "pokec_dataset/parametric_params"
    y_label_path = os.path.join(param_folder, f"y_label{total_n}.pk")
    y_unlabel_path = os.path.join(param_folder, f"y_unlabel_label{total_n}.pk")

    if os.path.exists(y_label_path) and os.path.exists(y_unlabel_path):
        with open(y_label_path, "rb") as f:
            y_label = pickle.load(f)
        with open(y_unlabel_path, "rb") as f:
            y_unlabel_label = pickle.load(f)
        return y_label, y_unlabel_label

    y_label = compute_score(df_labeled, target_column, args)
    y_unlabel_label = compute_score(df_unlabeled, target_column, args)

    with open(y_label_path, "wb") as f:
        pickle.dump(y_label, f)
    with open(y_unlabel_path, "wb") as f:
        pickle.dump(y_unlabel_label, f)

    return y_label, y_unlabel_label


def load_or_compute_features(df_labeled, df_unlabeled, target_column, include_graph_features, args):
    labeled_path = f"pokec_dataset/labeled_feature_matrix_{target_column}_{include_graph_features}.pk"
    unlabeled_path = f"pokec_dataset/unlabeled_feature_matrix_{target_column}_{include_graph_features}.pk"

    if os.path.exists(labeled_path) and os.path.exists(unlabeled_path):
        with open(labeled_path, "rb") as f:
            X_features_labeled = pickle.load(f)
        with open(unlabeled_path, "rb") as f:
            X_features_unlabeled = pickle.load(f)
        print("Feature matrix shape:", X_features_labeled.shape)
        return X_features_labeled, X_features_unlabeled

    filtered_textual_features = [text for text in textual_features if text != target_column]
    numerical_features_extended = numerical_features.copy()

    df_labeled[numerical_features_extended] = df_labeled[numerical_features_extended].apply(
        pd.to_numeric, errors="coerce"
    )
    df_labeled[categorical_features] = df_labeled[categorical_features].astype(str)
    df_labeled[filtered_textual_features] = df_labeled[filtered_textual_features].astype(str)

    df_unlabeled[numerical_features_extended] = df_unlabeled[numerical_features_extended].apply(
        pd.to_numeric, errors="coerce"
    )
    df_unlabeled[categorical_features] = df_unlabeled[categorical_features].astype(str)
    df_unlabeled[filtered_textual_features] = df_unlabeled[filtered_textual_features].astype(str)

    X_features_labeled = extract_features(
        df_labeled, args, filtered_textual_features, numerical_features_extended
    )
    X_features_unlabeled = extract_features(
        df_unlabeled, args, filtered_textual_features, numerical_features_extended
    )

    with open(labeled_path, "wb") as f:
        pickle.dump(X_features_labeled, f)
    with open(unlabeled_path, "wb") as f:
        pickle.dump(X_features_unlabeled, f)

    return X_features_labeled, X_features_unlabeled


def load_or_create_platform_sus(agent_num: int, param_folder: Path):
    platform_file_path = param_folder / f"hetero_platform_sus{agent_num}.pkl"
    if platform_file_path.exists():
        with open(platform_file_path, "rb") as f:
            return pickle.load(f)

    # Fallback only when pre-generated values are absent.
    platform_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
    with open(platform_file_path, "wb") as f:
        pickle.dump(platform_sus, f)
    return platform_sus


def load_or_create_peer_sus(agent_num: int, param_folder: Path):
    peer_file_path = param_folder / f"hetero_peer_sus{agent_num}.pkl"
    if peer_file_path.exists():
        with open(peer_file_path, "rb") as f:
            return pickle.load(f)

    peer_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
    with open(peer_file_path, "wb") as f:
        pickle.dump(peer_sus, f)
    return peer_sus


def main():
    args = parse_args()
    param_folder = Path("pokec_dataset/parametric_params")
    param_folder.mkdir(exist_ok=True)
    results_folder = Path("pokec_dataset/results")
    results_folder.mkdir(exist_ok=True)

    target_column = "relation_to_smoking"
    include_graph_features = False

    df, network_lcc = load_profiles_and_graph(target_column)

    if include_graph_features:
        df = add_graph_features(df, graph_path=f"pokec_dataset/lcc_graph_{target_column}.pk")

    n = int(len(df) * 0.8)
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()

    y_label, y_unlabel_label = load_or_compute_scores(
        df_labeled, df_unlabeled, target_column, args, len(df)
    )

    X_features_labeled, X_features_unlabeled = load_or_compute_features(
        df_labeled, df_unlabeled, target_column, include_graph_features, args
    )

    innate_original = np.array(y_label + y_unlabel_label, dtype=float)
    agent_num = len(innate_original)

    platform_sus = load_or_create_platform_sus(agent_num, param_folder)
    peer_sus = load_or_create_peer_sus(agent_num, param_folder)
    steer_strength = np.zeros(agent_num)

    rng = np.random.default_rng(2026)
    stubborn_idx = int(rng.integers(low=n, high=agent_num))

    peer_sus_modified = peer_sus.copy()
    # in paper alpha=0 is equivalent to set it to be 1.0 here
    peer_sus_modified[stubborn_idx] = 1.0

    innate_modified = innate_original.copy()
    labeled_override = max(1, int(0.1 * n))
    override_idx = rng.choice(np.arange(n), size=labeled_override, replace=False)
    innate_modified[override_idx] = 1.0

    retrain_T = 30
    fj_K = 100
    nodelist = df["user_id"].values


    labels = {"mean": "Mean", "perfect": "Perfect", "ridge": "OLS", "neural_net": "MLP"}
    # "perfect", "mean", "ridge", 
    for model_name in ["neural_net"]:
        if os.path.exists(results_folder / f"{model_name}_sl_modified_stubborn_whole_record{retrain_T}.pk"):
            print("Results already exist. Skipping simulation.")
            with open(results_folder / f"{model_name}_sl_modified_stubborn_whole_record{retrain_T}.pk", "rb") as f:
                modified_record = pickle.load(f)
            with open(results_folder / f"{model_name}_sl_original_whole_record{retrain_T}.pk", "rb") as f:
                baseline_record = pickle.load(f)
            with open(param_folder / f"stubborn_unlabeled_node_{agent_num}.pkl", "rb") as f:
                stubborn_idx = pickle.load(f)
            
        else:
        # Baseline: unmodified innate opinions for the horizontal reference line.
            baseline_record = run_simulation(
                network=network_lcc,
                nodelist=nodelist,
                platform_params=platform_sus,
                peer_params=peer_sus_modified,
                steering_params=steer_strength,
                steering_vector=None,
                retrain_steps=retrain_T,
                fj_steps=fj_K,
                x_star=innate_original,
                policy="sl",
                model_name=model_name,
                X_features_labeled=X_features_labeled,
                X_features_unlabeled=X_features_unlabeled,
            )

            # Intervention: peer-stubborn unlabeled agent and 10% labeled opinions set to 1.
            modified_record = run_simulation(
                network=network_lcc,
                nodelist=nodelist,
                platform_params=platform_sus,
                peer_params=peer_sus_modified,
                steering_params=steer_strength,
                steering_vector=None,
                retrain_steps=retrain_T,
                fj_steps=fj_K,
                x_star=innate_modified,
                policy="sl",
                model_name=model_name,
                X_features_labeled=X_features_labeled,
                X_features_unlabeled=X_features_unlabeled,
            )

            with open(results_folder / f"{model_name}_sl_modified_stubborn_whole_record{retrain_T}.pk", "wb") as f:
                pickle.dump(modified_record, f)
            with open(results_folder / f"{model_name}_sl_original_whole_record{retrain_T}.pk", "wb") as f:
                pickle.dump(baseline_record, f)
            with open(param_folder / f"stubborn_unlabeled_node_{agent_num}.pkl", "wb") as f:
                pickle.dump(stubborn_idx, f)
            
        x = np.arange(retrain_T + 1)
        stubborn_path = modified_record[stubborn_idx, :]
        # original_equilibrium = baseline_record[stubborn_idx, -1]
        original_path = baseline_record[stubborn_idx, :]

        plt.figure()
        plt.plot(
            x,
            stubborn_path,
            # marker="o",
            # markersize=3,
            # linewidth=1.8,
            color="tab:blue",
            label=labels[model_name] + r" $\tilde{x}^*$",
        )
        plt.plot(
            x,
            original_path,
            # marker="o",
            # markersize=3,
            # linewidth=1.8,
            color="tab:orange",
            label=labels[model_name] + r" $x^*$",
        )
        # plt.hlines(
        #     y=original_equilibrium,
        #     xmin=0,
        #     xmax=retrain_T,
        #     linestyle="--",
        #     linewidth=1.5,
        #     color="tab:red",
        #     label="Original equilibrium (unmodified innate opinions)",
        # )

        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.ylabel(r"Opinion $(x_{ex}^{(t)})_q$", fontsize=18)
        plt.xticks(np.arange(0,retrain_T + 1, 5), fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="best", frameon=False, fontsize=15)

        out_path = param_folder / f"{model_name}_parametric_sl_retrain_steps_stubborn_unlabeled.pdf"
        plt.savefig(out_path, bbox_inches="tight")

        print(f"stubborn unlabeled index: {stubborn_idx} (global index)")
        print(f"modified 10% labeled count: {labeled_override}")
        print(f"saved figure: {out_path}")


if __name__ == "__main__":
    main()
