# -*- coding: utf-8 -*-
# type: ignore
"""# Resources:

*   https://github.com/haosulab/ManiSkill/tree/main
*   https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
*   http://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html
"""

import os
from datetime import datetime

from typing import Union
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

torch.manual_seed(42)

DS_PATH = "/home/massimiliano/.maniskill/demos/PickCube-v1/motionplanning/"

H5_PATH = DS_PATH + "trajectory.h5"
REPLAYED_H5_PATH = DS_PATH + "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"

JS_PATH = DS_PATH + "trajectory.json"
REPLAYED_JS_PATH = DS_PATH + "trajectory.state.pd_ee_delta_pos.physx_cpu.json"

CHECKPOINT_PATH = "gatautoencoder_checkpoint_E50_2026-03-06T16:28:41.304114.pth"

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 32
HIDDEN_CHANNELS = 32
LATENT_CHANNELS = 16

# Phase 1: Scene Graph Engineering

# Data Extraction:
"""
Parse the privileged states from the IL dataset (e.g., object XYZ, bounding boxes, gripper pose).
"""

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
    """

    def __init__(
        self, dataset_file: str, load_count=-1, success_only: bool = False, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
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
                self.success = common.to_tensor(self.terminated, device=device)
            if self.fail is not None:
                self.fail = common.to_tensor(self.truncated, device=device)

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


dataset = ManiSkillTrajectoryDataset(REPLAYED_H5_PATH)
print(
    f"""Dataset lenght: {
        len(dataset)
    }, i.e., number of episodes/trajectories * frames per episode (-> (priv_state, observation, action) 3-ple for training later"""
)


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


# Actors: 13 dimensions for position + quaternon + velocity + angular_velocity
# Articulations: 13 dimensions for position + quaternon + velocity + angular_velocity
#   Note:   articulations is not intended as 7 actors (arms) + 2 (gripper hands), which would lead to (7+2)*13 dimensions
#
#           When two objects are joined by a hinge (a revolute joint), they lose almost all their relative freedom.
#           A free-floating actor has 6 DOFs (3 translation, 3 rotation). Once you bolt it to another actor with a hinge,
#           it only has 1 DOF relative to its parent—it can only rotate around one axis.
#
#           So we rather reduce coordinates of such hinged actors by taking a root (usually the 000 coordinate or the base)
#           bringing 13 dimensions (just as before) and joint position (1) + joint velocity (1) for each hinged actor (7+2)
#           so we get 13 + 9*(2) = 13 + 18 = 31
print_dict_tree(dataset[999])
print()
print("-" * 64)
print()

# Graph Construction:
"""
Nodes: Objects and gripper(s) with spatial features.
Edges: Spatial relationships defined by Euclidean distances.
Implementation: Build a preprocessing script to convert flat state vectors into graph structures (adjacency matrices + feature tensors).
"""

# Table is always the same thorugh all episodes, and for all frames of the episode
# Goal instead may change between episodes, but within the same is constant
# Cube always change obv, articulations as well
#
# Now, a couple of considerations:
#   1.  Even if some objects are constant, it doesnt mean they are useless; in a graph they can still
#       serve as relative "anchor" positions, which would actually make the learning generalize better into
#       other settings (informing about its "new" base/anchor, e.g. the robot could be ancored on the table, or the wall, or the floor)
#   2.  There is a thing called Proprioception, which is the capacity for the robot to understand where itself is in space
#       so informing it about its own presence of arms (othern than the hand itself) would be useful anyway, even if we have PID modules
#       which would compute these automtically based on the action sent to move the arm (indeed, knowing where its own base/anchor is, is Proprioception aswell!!)
#
# So a nice graph could be made of nodes being goal, hand1, hand2, table, anchor, cube (eventually the other arm joints for better Proprioception)
# each node would carry its own dedicated values in the dataset
#
# Then we make a complete graph with euclidean distance as weight for the edges, however we could cut distances below a threshold
# and enforce our own edges/non-edges, e.g., we could make the table node connected to the anchor only for simplicity
#
# Note sill that informing our robot about goal and the arm/hand positions is NOT data leaking, as we are not
# informing it about which actions to take! We are only tellig it that the goal is specifically there in space, and its own body is somewhere else
#


# TODO: I can also add other nodes and make a more complex graph, as well adding euclidean distance as edge weights
def build_graph(idx):
    priv_states = dataset[idx]["priv_states"]

    cube_xyz = priv_states["actors"]["cube"][:3]
    goal_xyz = priv_states["actors"]["goal_site"][:3]
    hand_xyz = priv_states["articulations"]["panda"][:3]

    # Append One-Hot Identity: [X, Y, Z, is_cube, is_goal, is_hand]
    cube_x = np.concatenate([cube_xyz, [1, 0, 0]])
    goal_x = np.concatenate([goal_xyz, [0, 1, 0]])
    hand_x = np.concatenate([hand_xyz, [0, 0, 1]])

    # Nodes (X) now has shape [3, 6]
    x = torch.tensor(np.array([cube_x, goal_x, hand_x]), dtype=torch.float)

    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
    )

    return Data(x=x, edge_index=edge_index)


print(build_graph(999))


# Phase 2: Representation Learning (The GAE)

# Architecture:
"""
Design a GNN-based Auto-Encoder (GCN or GAT).
"""
# Bottleneck:
"""
Compress the graph into a fixed-length latent vector z.
"""
# Validation:
"""
Ensure the embedding z is expressive enough to reconstruct the scene geometry accurately.
"""


# TODO: I should also implment a second head for reconstructing edge_index or A
class GATAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, heads=4):
        super().__init__()
        # ENCODER: Maps 6 dims -> hidden
        self.encoder_conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.encoder_conv2 = GATConv(
            hidden_channels * heads, latent_channels, heads=1, concat=False
        )

        # DECODER: Takes the GLOBAL latent vector + Node Identity -> Reconstructs XYZ
        # Input to decoder: latent_channels (e.g., 16) + 3 (identity) = 19
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels + 3, hidden_channels),
            nn.ReLU(),
            # We only want to predict the 3 XYZ coords
            nn.Linear(hidden_channels, 3),
        )

    def encode(self, x, edge_index, batch):
        x = self.encoder_conv1(x, edge_index)
        x = F.elu(x)
        node_z = self.encoder_conv2(x, edge_index)

        # THE BOTTLENECK: Pool all nodes in the graph into ONE vector z
        global_z = global_mean_pool(node_z, batch)
        return global_z

    def forward(self, data):
        # 1. Encode to a single scene vector [BatchSize, latent_channels]
        global_z = self.encode(data.x, data.edge_index, data.batch)

        # 2. To decode, we expand the global vector back to all nodes
        z_expanded = global_z[data.batch]

        # 3. Give the decoder the Scene Context (z) AND the Node Identity (last 3 columns of x)
        identities = data.x[:, 3:]
        dec_input = torch.cat([z_expanded, identities], dim=-1)

        # 4. Predict XYZ
        reconstructed_xyz = self.decoder(dec_input)

        return reconstructed_xyz, global_z


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out, _ = model(data)

        # Target is ONLY the first 3 columns (XYZ)
        target_xyz = data.x[:, :3]
        loss = F.mse_loss(out, target_xyz)

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
        out, _ = model(data)
        # Target is ONLY the first 3 columns (XYZ)
        target_xyz = data.x[:, :3]
        loss = F.mse_loss(out, target_xyz)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


# Training and Validation Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# Assuming you've created a list of Data objects using your build_graph function
data_list = [build_graph(i) for i in range(len(dataset))]
train_loader = DataLoader(data_list[:800], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(data_list[800:], batch_size=BATCH_SIZE)

in_channels = data_list[0].x.shape[1]

model = GATAutoencoder(
    in_channels=in_channels,
    hidden_channels=HIDDEN_CHANNELS,
    latent_channels=LATENT_CHANNELS,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from epoch {start_epoch}. Skipping training.")
else:
    # Proceed with training from scratch
    start_epoch = 0
    print("No checkpoint found. Starting training from scratch.")

# Simple Loop
trained = False
for epoch in range(start_epoch, EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)
    if epoch % 1 == 0:
        print(
            f"Epoch {epoch:03d}, Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}"
        )
    trained = True

if trained:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "hyperparameters": {
            "in_channels": in_channels,
            "hidden_channels": HIDDEN_CHANNELS,
            "latent_channels": LATENT_CHANNELS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
        },
    }

    # Get current timestamp (seconds since epoch)
    now = datetime.now()
    torch.save(checkpoint, f"gatautoencoder_checkpoint_E{epoch}_{now.isoformat()}.pth")
    print(f"Saved checkpoint at epoch {epoch} with timestamp {now.isoformat()}")

# Phase 3: Dataset Augmentation

# Encoding:
"""
Pass the entire IL dataset through the frozen GAE encoder.
"""


def get_embedding(idx):
    return model.encode(
        data_list[idx].x.to(device),
        data_list[idx].edge_index.to(device),
        torch.zeros(data_list[idx].x.shape[0], dtype=torch.long).to(device),
    )


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, model, device):
        self.base_dataset = base_dataset
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        # Build graph from priv_states
        graph = build_graph(idx)
        graph = graph.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.encode(
                graph.x,
                graph.edge_index,
                torch.zeros(graph.x.shape[0], dtype=torch.long, device=self.device),
            )
        sample["embedding"] = embedding.cpu().numpy()
        return sample


embedding_dataset = EmbeddingDataset(dataset, model, device)

# Storage:
"""
Save the resulting embeddings as a new key in the dataset (HDF5/Zarr).
"""

# TODO: later, unless it becomes to heavy to compute everything

# Smoothing:
"""
Verify temporal consistency of embeddings across trajectory frames.
"""

episode_lengths = []
with h5py.File(REPLAYED_H5_PATH, "r") as f:
    for traj_name in f.keys():
        num_frames = len(f[traj_name]["env_states"]["actors"]["cube"])
        episode_lengths.append(num_frames)


def compute_trajectory_embeddings_similarity(trajectory_embeddings):
    # trajectory_embeddings shape: [T, 16]
    z_t = trajectory_embeddings[:-1]
    z_next = trajectory_embeddings[1:]

    # Calculate similarity between adjacent frames
    sim = F.cosine_similarity(z_t, z_next, dim=-1)
    dissim = F.cosine_similarity(
        trajectory_embeddings[0], trajectory_embeddings[-1], dim=-1
    )

    return sim, dissim


def check_temporal_consistency(episode_idx):
    episodes = np.array(
        [embedding_dataset[i]["embedding"] for i in range(episode_lengths[episode_idx])]
    ).squeeze(1)

    similarity_scores, dissimilarity_score = compute_trajectory_embeddings_similarity(
        torch.tensor(episodes)
    )

    print(f"Mean Temporal Similarity: {similarity_scores.mean().item():.4f}")
    print(f"Min Temporal Similarity: {similarity_scores.min().item():.4f}")
    print(f"Max Temporal Similarity: {similarity_scores.max().item():.4f}")
    print(f"Dissimilarity (First vs Last): {dissimilarity_score.item():.4f}")


for i in range(1000):
    print("Episode", i)
    check_temporal_consistency(i)
    print()

# Phase 4: Policy Training & Evaluation


"""
State Input: Train an IL policy (BC) using a concatenated state:

Proprioception: (Joint positions, gripper state).

Latent State: The GAE scene embedding z.

Benchmarking: Compare the success rate of the Graph-State Policy against a baseline trained on raw, flat privileged coordinates.
"""
