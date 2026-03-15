# -*- coding: utf-8 -*-
# type: ignore
"""# Resources:

*   https://github.com/haosulab/ManiSkill/tree/main
*   https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
*   http://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html
"""

import shutil
import json
import argparse
from typing import Union
from pathlib import Path
from tqdm import tqdm

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Seeding


seed_everything(42)

# CUDA avaliability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NOTE: Should also manage physix_cuda but there were some issues with arrays/tensors
print(f"Working on {device}")

# For proper path generation
home = Path.home()
cwd = Path.cwd()
script_location = Path(__file__).resolve().parent

# Suppose a unix-like environment (I have no idea about Windows)
DS_PATH = home / ".maniskill/demos/PickCube-v1/motionplanning/"

# Paths to the datasets (H5)
H5_PATH = DS_PATH / "trajectory.h5"
REPLAYED_H5_PATH = DS_PATH / "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
EMBEDDINGS_H5_PATH = REPLAYED_H5_PATH.with_suffix(".embeddings.h5")

# And related json metadata
JS_PATH = DS_PATH / "trajectory.json"
REPLAYED_JS_PATH = DS_PATH / "trajectory.state.pd_ee_delta_pos.physx_cpu.json"
EMBEDDINGS_JS_PATH = REPLAYED_JS_PATH.with_suffix(".embeddings.json")

# Importable from outside as default
GAT_CHECKPOINT_PATH = script_location / "gatautoencoder_best.pth"

# Graph parameters
NUM_NODES = 5  # Cube, Goal, Table, Robot Base, Hand
IN_CHANNELS = 6  # XYZ + Bounding Box Size

# Hyperparameters
GAT_BATCH_SIZE = 64
GAT_HIDDEN_CHANNELS = 128
GAT_LATENT_CHANNELS = 4
GAT_ATTENTION_HEADS = 2


def print_dict_tree(data, indent=""):
    """
    Recursively prints a dictionary as a tree.
    Prints .shape and .dtype for atomic elements that possess them.
    """
    items = list(data.items())
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "

        # Branch: If the value is another dictionary, recurse
        if isinstance(value, dict):
            print(f"{indent}{branch}{key}")
            new_indent = indent + ("    " if is_last else "│   ")
            print_dict_tree(value, new_indent)

            # Leaf: If the value has a shape (and potentially a dtype)
        elif hasattr(value, "shape"):
            # Get dtype if it exists, otherwise leave empty
            dtype_str = f", dtype={value.dtype}" if hasattr(value, "dtype") else ""
            print(f"{indent}{branch}{key}: shape={value.shape}{dtype_str}")

            # Leaf: Basic types
        else:
            print(f"{indent}{branch}{key}: {type(value).__name__}")

    # loads h5 data into memory for faster access


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device

    Reference: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html    #pytorch
    """

    def __init__(
        self, dataset_file: str, load_count=-1, success_only: bool = False, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.with_suffix(".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs = None
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only:
                assert "success" in eps, (
                    "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                )
                if not eps["success"]:
                    continue
            trajesctory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajesctory)
            eps_len = len(trajectory["actions"])

            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            if eps_id == 0:
                self.obs = obs
            else:
                self.obs = common.append_dict_array(self.obs, obs)

            priv_states = common.index_dict_array(
                trajectory["env_states"], slice(eps_len)
            )
            if eps_id == 0:
                self.priv_states = priv_states
            else:
                self.priv_states = common.append_dict_array(
                    self.priv_states, priv_states
                )

            self.actions.append(trajectory["actions"])
            self.terminated.append(trajectory["terminated"])
            self.truncated.append(trajectory["truncated"])

            # handle data that might optionally be in the trajectory
            if "rewards" in trajectory:
                if self.rewards is None:
                    self.rewards = [trajectory["rewards"]]
                else:
                    self.rewards.append(trajectory["rewards"])
            if "success" in trajectory:
                if self.success is None:
                    self.success = [trajectory["success"]]
                else:
                    self.success.append(trajectory["success"])
            if "fail" in trajectory:
                if self.fail is None:
                    self.fail = [trajectory["fail"]]
                else:
                    self.fail.append(trajectory["fail"])

        self.actions = np.vstack(self.actions)
        self.terminated = np.concatenate(self.terminated)
        self.truncated = np.concatenate(self.truncated)

        if self.rewards is not None:
            self.rewards = np.concatenate(self.rewards)
        if self.success is not None:
            self.success = np.concatenate(self.success)
        if self.fail is not None:
            self.fail = np.concatenate(self.fail)

        def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x

            # uint16 dtype is used to conserve disk space and memory
            # you can optimize this dataset code to keep it as uint16 and process that
            # dtype of data yourself. for simplicity we simply cast to a int32 so
            # it can automatically be converted to torch tensors without complaint

        self.obs = remove_np_uint16(self.obs)

        if device is not None:
            self.actions = common.to_tensor(self.actions, device=device)
            self.obs = common.to_tensor(self.obs, device=device)
            self.terminated = common.to_tensor(self.terminated, device=device)
            self.truncated = common.to_tensor(self.truncated, device=device)
            if self.rewards is not None:
                self.rewards = common.to_tensor(self.rewards, device=device)
            if self.success is not None:
                self.success = common.to_tensor(self.success, device=device)
            if self.fail is not None:
                self.fail = common.to_tensor(self.fail, device=device)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = common.to_tensor(self.actions[idx], device=self.device)
        obs = common.index_dict_array(self.obs, idx, inplace=False)
        priv_states = common.index_dict_array(self.priv_states, idx, inplace=False)

        res = dict(
            action=action,
            priv_states=priv_states,
            obs=obs,
            # action=action,
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
        )
        if self.rewards is not None:
            res.update(reward=self.rewards[idx])
        if self.success is not None:
            res.update(success=self.success[idx])
        if self.fail is not None:
            res.update(fail=self.fail[idx])
        return res


def build_graph(dataset, idx):
    priv_states = dataset[idx]["priv_states"]
    obs = dataset[idx]["obs"]

    # Extract XYZ for the nodes
    cube_xyz = priv_states["actors"]["cube"].squeeze()[:3]
    goal_xyz = priv_states["actors"]["goal_site"].squeeze()[:3]
    table_xyz = priv_states["actors"]["table-workspace"].squeeze()[:3]
    base_xyz = priv_states["articulations"]["panda"].squeeze()[:3]

    # TCP is not available in the SAPIENS data, so we need to retrive it from obs
    # NOTE: Maybe this one not?
    hand_xyz = obs.squeeze()[19:22]

    # Bounding boxes
    # Reference:
    #   https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py
    #   https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/scene_builder/table/scene_builder.py
    cube_box = np.array(
        [0.04, 0.04, 0.04]
    )  # see cube_half_size variable in pick_cube_cfgs.py, which is set to 0.02
    goal_box = np.array(
        [0.05, 0.05, 0.05]
    )  # goal_tresh is 0.025 by default, see pick_cube_cfgs.py
    table_box = np.array(
        [1.21, 2.42, 0.92]
    )  # always set so the surface is at z=0, see aabb and below rows in table/scene_builder.py
    base_box = np.array([0.20, 0.20, 0.20])  # educated guess(?)
    hand_box = np.array([0.10, 0.05, 0.10])  # educated guess(?)

    # Concatenate XYZ with Identities -> Shape: [5 nodes, 8 features]
    # NOTE: Most of these nodes are actually static, this may lead to a really good model later since it understands
    # that it can achive low MSE by simply memorizing the table, base and goal positions.
    # Maybe I should just ignore these? ask Davide
    # IDEA: I could first train the whole model and then fine tune it on the non-static nodes specifically
    nodes_list = [
        np.concatenate([cube_xyz, cube_box]),
        np.concatenate([goal_xyz, goal_box]),
        np.concatenate([table_xyz, table_box]),
        np.concatenate([base_xyz, base_box]),
        np.concatenate([hand_xyz, hand_box]),
    ]
    x = torch.tensor(np.array(nodes_list), dtype=torch.float)

    # Generate arbitrary edges
    sparse_edges = [
        (4, 0),
        (0, 4),  # Hand <-> Cube
        (4, 1),
        (1, 4),  # Hand <-> Goal
        (4, 3),
        (3, 4),  # Hand <-> Base
        (0, 1),
        (1, 0),  # Cube <-> Goal
        (0, 2),
        (2, 0),  # Cube <-> Table
    ]
    edge_index = torch.tensor(sparse_edges, dtype=torch.long).t().contiguous()

    # Calculate Euclidean Distances for Edge Weights
    # Get the XYZ coordinates for the source (row) and target (col) of each edge
    row, col = edge_index
    src_xyz = x[row, :3]
    dst_xyz = x[col, :3]

    # Calculate the L2 Norm (Euclidean distance) between them
    # NOTE: I could actually leverage those 6 last elements in obs_extras to improve this
    distances = torch.linalg.vector_norm(src_xyz - dst_xyz, ord=2, dim=1)
    distances = distances.view(-1, 1)

    # Return the graph
    return Data(x=x, edge_index=edge_index, edge_attr=distances)


class GATAutoencoder(nn.Module):
    def __init__(
        self,
        number_of_nodes,
        in_channels,
        hidden_channels,
        latent_channels,
        heads,
    ):
        super().__init__()
        self.number_of_nodes = number_of_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.heads = heads

        # ENCODER: graph -> hidden -> latent
        self.encoder_conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, edge_dim=1
        )
        self.encoder_conv2 = GATConv(
            hidden_channels * heads, latent_channels, heads=1, concat=False, edge_dim=1
        )

        # FEATURES DECODER: latent vector -> reconstructs features
        self.features_decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

        # TOPOLOGY DECODER: pair of latent vectors -> reconstructs weights
        self.topology_decoder = nn.Sequential(
            nn.Linear(
                latent_channels * 2, hidden_channels
            ),  # Takes 2 concatenated nodes
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),  # Outputs a single scalar distance
        )

    def encode(self, data):
        x = self.encoder_conv1(data.x, data.edge_index, data.edge_attr)
        x = F.elu(x)
        node_z = self.encoder_conv2(x, data.edge_index, data.edge_attr)

        return node_z

    def forward(self, data):
        # ENCODE
        node_z = self.encode(data)

        # DECODE FEATURES
        reconstructed_feat = self.features_decoder(node_z)

        # DECODE TOPOLOGY
        row, col = data.edge_index
        # Concatenate source and destination embeddings
        z_pairs = torch.cat([node_z[row], node_z[col]], dim=-1)
        topology_pred = self.topology_decoder(z_pairs)

        # FLATTEN GLOBAL Z FOR THE BC POLICY
        # Reshapes from [320, 8] -> [64, 40]
        global_z = node_z.view(data.num_graphs, self.number_of_nodes * node_z.size(-1))

        return reconstructed_feat, topology_pred, global_z


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out_feat, out_topo, _ = model(data)

        # Target tensors
        target_feat = data.x
        target_topo = data.edge_attr

        # Reshape from [320, F] -> [64, 5, F]
        out_feat_reshaped = out_feat.view(data.num_graphs, model.number_of_nodes, -1)
        target_feat_reshaped = target_feat.view(
            data.num_graphs, model.number_of_nodes, -1
        )

        # Compute MSE for features
        # 0: Cube, 4: Hand vs. 1: Goal, 2: Table, 3: Base
        loss_dynamic = F.mse_loss(
            out_feat_reshaped[:, [0, 4], :], target_feat_reshaped[:, [0, 4], :]
        )
        loss_static = F.mse_loss(
            out_feat_reshaped[:, [1, 2, 3], :], target_feat_reshaped[:, [1, 2, 3], :]
        )
        loss_features = (10.0 * loss_dynamic) + loss_static

        # Compute topology MSE (weights)
        loss_topology = F.mse_loss(out_topo, target_topo)

        # Combine
        loss = loss_features + loss_topology

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)

        # Forward pass
        out_feat, out_topo, _ = model(data)

        # Target tensors
        target_feat = data.x
        target_topo = data.edge_attr

        # Reshape from [320, F] -> [64, 5, F]
        out_feat_reshaped = out_feat.view(data.num_graphs, model.number_of_nodes, -1)
        target_feat_reshaped = target_feat.view(
            data.num_graphs, model.number_of_nodes, -1
        )

        # Compute MSE for features
        # 0: Cube, 4: Hand vs. 1: Goal, 2: Table, 3: Base
        loss_dynamic = F.mse_loss(
            out_feat_reshaped[:, [0, 4], :], target_feat_reshaped[:, [0, 4], :]
        )
        loss_static = F.mse_loss(
            out_feat_reshaped[:, [1, 2, 3], :], target_feat_reshaped[:, [1, 2, 3], :]
        )
        loss_features = (10.0 * loss_dynamic) + loss_static

        # Compute topology MSE (weights)
        loss_topology = F.mse_loss(out_topo, target_topo)

        # Combine MSEs
        loss = loss_features + loss_topology

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def get_all_episode_lengths(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [ep["elapsed_steps"] for ep in data["episodes"]]


def compute_and_store_embeddings(model, base_h5_path, output_h5_path):
    """
    Copies the original HDF5 file and appends embeddings for each frame under each  group.
    """
    if base_h5_path.exists():
        shutil.copy(base_h5_path, output_h5_path)
        base_json_path = base_h5_path.with_suffix(".json")
        output_json_path = base_json_path.with_suffix(".embeddings.json")
        print("Copied HDF5 data to new file for embedding storage.")
    else:
        raise FileNotFoundError(f"Base HDF5 file not found at {base_h5_path}")

    if base_json_path.exists():
        shutil.copy(base_json_path, output_json_path)
        print(f"Copied JSON metadata to {output_json_path}.")
    else:
        print(f"Warning: Expected JSON metadata not found at {base_json_path}")

    with h5py.File(output_h5_path, "a") as f:
        global_frame_idx = 0
        trajectories = sorted(list(f.keys()), key=lambda x: int(x.split("_")[1]))
        for traj_name in tqdm(trajectories):  # Loop over episodes/trajectories
            traj_group = f[traj_name]
            num_frames = len(traj_group["actions"])
            embeddings = []
            for _ in range(
                num_frames
            ):  # Loop over frames within the episode/trajectory
                graph = build_graph(dataset, global_frame_idx)
                graph = graph.to(device)
                model.eval()
                with torch.no_grad():
                    emb = model.encode(graph)
                embeddings.append(emb.cpu().numpy().squeeze())
                global_frame_idx += 1
            embeddings = np.stack(embeddings)
            traj_group["env_states"].create_dataset("embeddings", data=embeddings)
    print(f"Embeddings computed and stored in {output_h5_path}")


def compute_trajectory_embeddings_similarity(trajectory_embeddings):
    # trajectory_embeddings shape: [T, GAT_LATENT_CHANNELS]
    z_t = trajectory_embeddings[:-1]
    z_next = trajectory_embeddings[1:]

    # Calculate similarity between adjacent frames
    sim_scores = F.cosine_similarity(z_t, z_next, dim=-1)

    return sim_scores


# NOTE: Could as well check via plots and other measures, but for now I go for this one here
def check_temporal_consistency(start_frame_idx, episode_length):
    episode = embeddings_dataset[start_frame_idx : start_frame_idx + episode_length]
    embeddings = torch.tensor(episode["priv_states"]["embeddings"])

    sim_scores = compute_trajectory_embeddings_similarity(embeddings)

    print(f"Mean Temporal Similarity: {sim_scores.mean().item():.4f}")


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="ManiSkill GAT-BC Training and Benchmarking"
    )
    parser.add_argument(
        "--gat-epochs", type=int, default=25, help="Number of GAT epochs (default: 10)"
    )
    parser.add_argument(
        "--gat-checkpoint",
        type=str,
        default="gatautoencoder_best.pth",
        help="GAT checkpoint filename (default: gatautoencoder_best.pth)",
    )

    args = parser.parse_args()

    GAT_EPOCHS = args.gat_epochs
    GAT_CHECKPOINT_PATH = script_location / args.gat_checkpoint
    SPLIT_RATIO = 0.8
    GAT_LR = 1e-4

    dataset = ManiSkillTrajectoryDataset(REPLAYED_H5_PATH)
    print_dict_tree(dataset[0])
    print(build_graph(dataset, 0))

    # Extract episodes lenghts from the json file and store it in a list
    episode_lengths = get_all_episode_lengths(REPLAYED_JS_PATH)

    # Beware of data leakage: I should split over episodes, not frames themselves
    # In a simulation, Frame 45 and Frame 46 of the same episode are 99.9% identical
    # If random splitting puts Frame 45 in your Train Set and Frame 46 in your Validation Set, your Validation MSE will drop to near zero
    # Train Set: Episodes 0 to 800 (contains all their frames)
    # Validation Set: Episodes 800 to 1000 (contains all their frames)
    # WARNING: We should do the same exact split for the BC policy later

    # Create a list of Data objects using build_graph function
    data_list = [build_graph(dataset, i) for i in range(len(dataset))]

    # Extract splitting index over dataset (remind dataset is a flatten sequence of episode's frame, so we need to reconstruct the frame that divides the 80/20 of episodes)
    episode_lengths = get_all_episode_lengths(JS_PATH)
    num_train_episodes = int(SPLIT_RATIO * len(episode_lengths))
    split_idx = sum(episode_lengths[:num_train_episodes])
    print(f"Splitting data for the GAT at {split_idx}")

    # Split the data
    train_dataset = data_list[:split_idx]
    val_dataset = data_list[split_idx:]

    # Move it to dataloaders
    train_loader = GeoDataLoader(train_dataset, batch_size=GAT_BATCH_SIZE, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=GAT_BATCH_SIZE)

    # Get input feature dimension from the first graph (should be 6 in our case: XYZ + One-Hot Identity)
    num_nodes = data_list[0].x.shape[0]
    in_channels = data_list[0].x.shape[1]

    assert (
        NUM_NODES == num_nodes,
        "Expected number of nodes does not match the one in the graph data",
    )
    assert (
        IN_CHANNELS == in_channels,
        "Expected input feature dimension does not match the one in the graph data",
    )

    # NOTE: another approach would be to directly train the whole BC + GAT model all togheter
    # Prepare the GNN, optmizer etc.
    model = GATAutoencoder(
        number_of_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=GAT_HIDDEN_CHANNELS,
        latent_channels=GAT_LATENT_CHANNELS,
        heads=GAT_ATTENTION_HEADS,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GAT_LR)

    # Check for existence of checkpoint to resume/skip training
    best_val_loss = float("inf")
    if GAT_CHECKPOINT_PATH.exists():
        # WARNING: You should also load the seed state if you want to have a perfect reproducibility
        checkpoint = torch.load(GAT_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(
            f"Loaded checkpoint from epoch {start_epoch} with best val loss: {best_val_loss:.4f}"
        )
    else:
        # Proceed with training from scratch
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")

    # Loop
    for epoch in range(start_epoch, GAT_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}, Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "hyperparameters": {
                    "in_channels": in_channels,
                    "hidden_channels": GAT_HIDDEN_CHANNELS,
                    "latent_channels": GAT_LATENT_CHANNELS,
                    "batch_size": GAT_BATCH_SIZE,
                    "lr": GAT_LR,
                    "epochs": GAT_EPOCHS,
                },
            }
            torch.save(checkpoint, GAT_CHECKPOINT_PATH)
            print(f"New best GAT model saved with Val MSE: {val_loss:.4f}")

    # Load the best model weights for later phases
    if GAT_CHECKPOINT_PATH.exists():
        print(f"Loading best GAT model from {GAT_CHECKPOINT_PATH}")
        checkpoint = torch.load(GAT_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])

    # *** Phase 3: Dataset Augmentation ***

    print("\n\n--- Phase 3: Dataset Augmentation ---")

    # Check for existence of the embeddings dataset, if not, compute and store it
    if not EMBEDDINGS_H5_PATH.exists():
        print(
            f"Embeddings H5 dataset {EMBEDDINGS_H5_PATH} not found. Computing and storing embeddings..."
        )
        compute_and_store_embeddings(model, REPLAYED_H5_PATH, EMBEDDINGS_H5_PATH)
    else:
        print(
            f"Embeddings H5 dataset {EMBEDDINGS_H5_PATH} already exists. Skipping embedding computation."
        )

    # Load on the same dataset, should work out of the box with the class made on top of the file
    embeddings_dataset = ManiSkillTrajectoryDataset(EMBEDDINGS_H5_PATH)
    print(f"""Dataset lenght: {len(dataset)}""")

    # Try it out
    print_dict_tree(embeddings_dataset[0])

    # Check temporal consistency for the first 5 episodes
    start_idx = 0
    for i in range(5):
        print(f"\nEpisode {i}")
        episode_len = episode_lengths[i]
        check_temporal_consistency(start_idx, episode_len)
        start_idx += episode_len
