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
from datetime import datetime
from tqdm import tqdm

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common
import gymnasium as gym

# Argument Parsing
parser = argparse.ArgumentParser(
    description="ManiSkill GAT-BC Training and Benchmarking"
)
parser.add_argument(
    "--gat-epochs", type=int, default=10, help="Number of GAT epochs (default: 10)"
)
parser.add_argument(
    "--bc-epochs", type=int, default=50, help="Number of BC epochs (default: 50)"
)
parser.add_argument(
    "--benchmark",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable benchmarking (default: True)",
)
parser.add_argument(
    "--render",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable rendering (default: True)",
)
parser.add_argument(
    "--gat-checkpoint",
    type=str,
    default="gatautoencoder_best.pth",
    help="GAT checkpoint filename (default: gatautoencoder_best.pth)",
)
parser.add_argument(
    "--bc-checkpoint",
    type=str,
    default="bc_res_policy_best.pth",
    help="BC checkpoint filename (default: bc_mlp_policy_best.pth)",
)
parser.add_argument(
    "--baseline-checkpoint",
    type=str,
    default="baseline_policy_best.pth",
    help="Baseline checkpoint filename (default: baseline_policy_best.pth)",
)
args = parser.parse_args()


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
now = datetime.now()

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

# These are to load/save checkpoints
GAT_CHECKPOINT_PATH = script_location / args.gat_checkpoint
BC_CHECKPOINT_PATH = script_location / args.bc_checkpoint
BASELINE_CHECKPOINT_PATH = script_location / args.baseline_checkpoint

BENCHMARK = args.benchmark
RENDER = args.render

# Generic Hyperparameters
SPLIT_RATIO = 0.8
NOISE_STD = 0.01

# GNN Hyperparameters
GAT_EPOCHS = args.gat_epochs
GAT_LR = 1e-4
GAT_BATCH_SIZE = 64
GAT_HIDDEN_CHANNELS = 64
GAT_LATENT_CHANNELS = 8
GAT_ATTENTION_HEADS = 4

# BC Hyperparameters
BC_EPOCHS = args.bc_epochs
BC_LR = 1e-3
BC_BATCH_SIZE = 1024
BC_ACTION_DIM = 4
BC_PROPRIO_DIM = 18
BC_GRIPPER_DIM = 8
BC_HIDDEN_DIM = 256
BC_RES_HIDDEN_DIM = 256
BC_RES_DROPOUT = 0.1

# %% *** Phase 1: Scene Graph Engineering ***

print("\n\n--- Phase 1: Data extraction ---")


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

    Reference: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html#pytorch
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


print_dict_tree(dataset[0])

# ** Privilege states **
#
# Note that `env_states` is simply a direct memory dump from the underlying SAPIENS engine
#
# Actors: 13 dimensions for position + quaternon + velocity + angular_velocity
# Articulations: 31 dimensions for position + quaternon + velocity + angular_velocity of root state + joints
#
#   Note:   articulations is not intended as 7 actors (arms) + 2 (gripper hands), which would lead to (7+2)*13 dimensions
#
#           When two objects are joined by a hinge (a revolute joint), they lose almost all their relative freedom.
#           A free-floating actor has 6 DOFs (3 translation, 3 rotation). Once you bolt it to another actor with a hinge,
#           it only has 1 DOF relative to its parent—it can only rotate around one axis.
#
#           So we rather reduce coordinates of such hinged actors by taking a root (usually the 000 coordinate or the base)
#           bringing 13 dimensions (just as before) and joint position (1) + joint velocity (1) for each hinged actor (7 for arm + 2 for gripper hand)
#           so we get 13 + 9*(2) = 13 + 18 = 31
#
#   Indexing of actors arrays
#
#           pose.p            -> 3  (x,y,z)
#           pose.q            -> 4  (quaternion w,x,y,z)
#           linear_velocity   -> 3
#           angular_velocity  -> 3
#
#   Indexing of articulations arrays
#
#           (ALL COSTANTS)
#           0:3   root position
#           3:7   root quaternion
#           7:10  root linear velocity
#           10:13 root angular velocity
#
#           (THESE CHANGE)
#           13:22 joint positions (9)
#           22:31 joint velocities (9)
#
#   References:
#       https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/utils/structs/actor.html
#       https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/utils/structs/articulation.html

# ** Observations **
#
# Note that SAPIENS tracks the world using the absolute minimum variables required to calculate collisions and gravity
# For this reason The "Hand" (or Tool Center Point - TCP) is not a physics object tracked.
# It is an imaginary geometric point floating between the two gripper fingers.
# To find out where the hand actually is in 3D space, you have to run Forward Kinematics—multiplying all 9 joint angles through a complex kinematic tree.
# ManiSkill's environment automatically runs that math and injects the result into the obs array, i.e.:
#
#   Indices,    Size,   Description
#
#   [0:9],      9,      qpos: Joint Angles (7 arm joints + 2 gripper fingers) -> SAME AS ENV_STATE
#   [9:18],     9,      qvel: Joint Velocities -> SAME AS ENV_STATE
#   []          0,      controller (often empty)
#   [18]        1,      is_grasped: bool
#   [19:22],    3,      tcp_pose (Position): X, Y, Z
#   [22:26],    4,      tcp_pose (Quaternion): W, X, Y, Z
#   [26:29]     3,      goal_pose (Position only): X, Y, Z
#   [29:42],    13,      Other Task-specific data, only included if "state" in obs_mode:
#                           "obj_pose": raw_pose,                         # Shape: (batch_size, 7) - Cube pose [x, y, z, qx, qy, qz, qw]
#                           "tcp_to_obj_pos": tensor,                     # Shape: (batch_size, 3) - Vector from TCP to cube
#                           "obj_to_goal_pos": tensor,                    # Shape: (batch_size, 3) - Vector from cube to goal
#
# Indeed, we can run:
#
# data = h5py.File(REPLAYED_H5_PATH, "r")
#
# Get Frame 0, then slice features 13 to 31
# print(data["traj_0"]["env_states"]["articulations"]["panda"][0, 13:31])
#
# Get Frame 0, then slice features 0 to 18
# print(data["traj_0"]["obs"][0, 0:18])
#
# And they will be identical
#
# Same goes for
#
# Get Frame 999, then slice over the cube pose (position and quaternon)
# data['traj_999']['env_states']['actors']['cube'][:5, :7]
#
# and
#
# Get Frame 999, then slice over observations for obj_pose
# data['traj_999']['obs'][:5, 29:36]
#
# As well as
#
# Get Frame 999, goal site XYZ
# data['traj_999']['env_states']['actors']['goal_site'][:5, :3]
#
# againsts
#
# Get Frame 999, goal_pose (no quaternon)
# data['traj_999']['obs'][:5, 26:29]
#
# Reference:
#
#       https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html#state-dict
#       https://maniskill.readthedocs.io/en/v3.0.0b10/_modules/mani_skill/envs/sapien_env.html#BaseEnv.get_obs
#       https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/agents/base_agent.html#BaseAgent.get_proprioception
#       https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/pick_cube.html#PickCubeEnv._get_obs_extra
#       See mani_skill/envs/tasks/tabletop/pick_cube.py#L132-L145 for what values are returned as extras in obs
#
#
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
# Preprocessing, dropout or batch norm will not be contemplated in the following, nor lr scheduler and other sophisticated tools
# as this is supposed to be some simple showcase; furthermore remind that all the controllers have a normalized
# action space ([-1, 1]) in Franka Emilia Panda robot, except arm_pd_joint_pos and arm_pd_joint_pos_vel


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

    # Build One-Hot Identities
    identities = np.eye(5)

    # Concatenate XYZ with Identities -> Shape: [5 nodes, 8 features]
    # NOTE: Most of these nodes are actually static, this may lead to a really good model later since it understands
    # that it can achive low MSE by simply memorizing the table, base and goal positions.
    # Maybe I should just ignore these? ask Davide
    # IDEA: I could first train the whole model and then fine tune it on the non-static nodes specifically
    nodes_list = [
        np.concatenate([cube_xyz, cube_box, identities[0]]),
        np.concatenate([goal_xyz, goal_box, identities[1]]),
        np.concatenate([table_xyz, table_box, identities[2]]),
        np.concatenate([base_xyz, base_box, identities[3]]),
        np.concatenate([hand_xyz, hand_box, identities[4]]),
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


# Try it out
print(build_graph(dataset, 0))


# %% *** Phase 2: Representation Learning ***

print("\n\n--- Phase 2: Representation Learning ---")


# NOTE: I could also implment a second head for reconstructing edge_index or adjacency matrix?
class GATAutoencoder(nn.Module):
    def __init__(
        self,
        number_of_nodes,
        in_channels,
        feature_size,
        hidden_channels,
        latent_channels,
        heads,
    ):
        super().__init__()
        # Number of features per node, identity excluded
        self.feature_size = feature_size
        # Number of nodes in the graph, i.e. size of the identity
        self.number_of_nodes = number_of_nodes

        # ENCODER: graph -> hidden -> latent
        self.encoder_conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, edge_dim=1
        )
        self.encoder_conv2 = GATConv(
            hidden_channels * heads, latent_channels, heads=1, concat=False, edge_dim=1
        )  # Since heads=1, concat=False is not really needed, but I put it for clarity

        # DECODER: latent vector + node identity -> reconstructs XYZ
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels + number_of_nodes, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, feature_size),
        )

    def encode(self, x, edge_index, edge_attr, batch):
        x = self.encoder_conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        node_z = self.encoder_conv2(x, edge_index, edge_attr)

        # Pool all nodes in the graph into z
        # NOTE: what if I just concat this?
        global_z = global_mean_pool(node_z, batch)
        return global_z

    def forward(self, data):
        # Encode to a single scene vector [BatchSize, latent_channels]
        global_z = self.encode(data.x, data.edge_index, data.edge_attr, data.batch)

        # To decode, we expand the global vector back to all nodes
        z_expanded = global_z[data.batch]

        # Give the decoder the encoded latent vector (z) and the node identities (values of x after the first 3 columns XYZ)
        identities = data.x[:, self.feature_size :]
        dec_input = torch.cat([z_expanded, identities], dim=-1)

        # Predict XYZ
        reconstructed_xyz = self.decoder(dec_input)

        return reconstructed_xyz, global_z


# Consider that proprioception will be already present in the input data to the IL model later,
# maybe it is redundant to pass it here?
#
# My answer: I dont think so, as features in the latent representation are not the same of the raw ones,
# they bring some extra information by interacting with other nodes in the graph!


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        # Load data on device and reset the gradients
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out, _ = model(data)

        # Loss (Target is only the first 3 columns (XYZ) since the remaning are the identities)
        target_xyz = data.x[:, : model.feature_size]
        loss = F.mse_loss(out, target_xyz)

        # Backward pass
        loss.backward()
        optimizer.step()

        # To have a proper loss printing, we weight this for the size of the batch
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    for data in loader:
        # Load data on device
        data = data.to(device)

        # Forward pass
        out, _ = model(data)

        # Loss (Target is only the first 3 columns (XYZ) since the remaning are the identities)
        target_xyz = data.x[:, : model.feature_size]
        loss = F.mse_loss(out, target_xyz)

        # To have a proper loss printing, we weight this for the size of the batch
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def get_all_episode_lengths(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [ep["elapsed_steps"] for ep in data["episodes"]]


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
feature_size = in_channels - num_nodes

# Prepare the GNN, optmizer etc.
model = GATAutoencoder(
    number_of_nodes=num_nodes,
    in_channels=in_channels,
    feature_size=feature_size,
    hidden_channels=GAT_HIDDEN_CHANNELS,
    latent_channels=GAT_LATENT_CHANNELS,
    heads=GAT_ATTENTION_HEADS,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=GAT_LR)

# Check for existence of checkpoint to resume/skip training
best_val_loss = float("inf")
if GAT_CHECKPOINT_PATH.exists():
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
    print(f"Epoch {epoch:03d}, Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")

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
                    emb = model.encode(
                        graph.x,
                        graph.edge_index,
                        graph.edge_attr,
                        torch.zeros(  # We do not have a PyG dataloader, so we need to make a dummy batch tensor: equivalently, a batch size of 1
                            graph.x.shape[0], dtype=torch.long, device=graph.x.device
                        ),
                    )
                embeddings.append(emb.cpu().numpy().squeeze())
                global_frame_idx += 1
            embeddings = np.stack(embeddings)
            traj_group["env_states"].create_dataset("embeddings", data=embeddings)
    print(f"Embeddings computed and stored in {output_h5_path}")


# Check for existence of the embeddings dataset, if not, compute and store it
if not EMBEDDINGS_H5_PATH.exists():
    print(
        f"Embeddings H5 dataset {EMBEDDINGS_H5_PATH} not found. Computing and storing embeddings..."
    )
    compute_and_store_embeddings(model, REPLAYED_H5_PATH, EMBEDDINGS_H5_PATH)

# Load on the same dataset, should work out of the box with the class made on top of the file
embeddings_dataset = ManiSkillTrajectoryDataset(EMBEDDINGS_H5_PATH)
print(
    f"""Dataset lenght: {
        len(dataset)
    }, i.e., number of episodes/trajectories * frames per episode (-> (priv_state, observation, action) 3-ple for training later"""
)

# Try it out
print_dict_tree(embeddings_dataset[0])


def compute_trajectory_embeddings_similarity(trajectory_embeddings):
    # trajectory_embeddings shape: [T, 16]
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


# Check temporal consistency for the first 5 episodes (cosine similarity over subsequent frames' embeddings)
start_idx = 0
for i in range(5):
    print(f"\nEpisode {i}")
    episode_len = episode_lengths[i]
    check_temporal_consistency(start_idx, episode_len)
    start_idx += episode_len
print()

# *** Phase 4: Policy Training & Evaluation ***

print("\n\n--- Phase 4: Policy Training & Evaluation ---")

# In ManiSkill examples, the PickCube-v1 task is addressed using three primary architectures:
#
#   1. Behavioral Cloning (BC)
#      A MLP with two hidden layers of 256 units and ReLU activations
#       - Trained on the whole dataset (no validation/test set)
#       - Trains on the whole obs group, then compares over actions via MSE
#       - Adam with 3e-4 LR
#       - Batch size of 1024
#       - 1 000 000 training iterations, 1 iteration = 1 batch
#      A second version uses a custom PlainConv visual encoder consisting of five convolutional layers (with ReLU and
#      MaxPool) to process RGB-D images. The resulting visual features are concatenated with the robot's state and passed to the MLP
#   2. Action Chunking with Transformers (ACT)
#   3. Diffusion Policy
#
#   See examples/baselines/bc, examples/baselines/act, and examples/baselines/diffusion_policy respectively,
#   with specific scripts like bc.py, train.py, and train_rgbd.py providing the configurations for the PickCube-v1 task


class ResBlock(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResNetGraphStateBCPolicy(nn.Module):
    def __init__(
        self, z_dim, proprio_dim, gripper_dim, action_dim, hidden_dim, dropout_p
    ):
        super().__init__()
        input_dim = z_dim + proprio_dim + gripper_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.res_stack = nn.Sequential(
            ResBlock(hidden_dim, dropout_p),
            ResBlock(hidden_dim, dropout_p),
            ResBlock(hidden_dim, dropout_p),
        )

        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, z, gripper, proprioception):
        x = torch.cat([z, gripper, proprioception], dim=-1)
        x = F.relu(self.input_layer(x))
        x = self.res_stack(x)
        return self.output_layer(x)


def train_bc_epoch(model, loader, optimizer, scheduler, device, noise_std=None):
    model.train()
    total_loss = 0
    for batch in loader:
        # Load data on device and reset the gradients
        z = batch["priv_states"]["embeddings"].to(device)
        gripper = batch["obs"][:, 18:26].to(device)  # is_grasped + tcp_pose
        proprio = batch["priv_states"]["articulations"]["panda"][:, 13:31].to(device)
        target_action = batch["action"].to(device)
        optimizer.zero_grad()

        # Data augmentation
        #
        # NOTE: Maybe this could lead to errors since TCP should be the consequence of the rest of the environment,
        # so adding noise to it may break the physical consistency of the data;
        # however, I think that if the noise is small enough, it should be fine and actually help the model to generalize better
        # anyway, ManiSkill benchmark doesnt do it
        if noise_std is not None:
            z = z + torch.randn_like(z) * noise_std
            gripper = gripper + torch.randn_like(gripper) * (noise_std * 0.5)
            proprio = proprio + torch.randn_like(proprio) * noise_std

        # Forward pass
        out = model(z, gripper, proprio)

        # Loss
        loss = F.mse_loss(out, target_action)

        # Backward pass
        loss.backward()
        optimizer.step()

        # To have a proper loss printing, we weight this for the size of the batch
        total_loss += loss.item() * z.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_bc(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        # Load data on device
        z = batch["priv_states"]["embeddings"].to(device)
        gripper = batch["obs"][:, 18:26].to(device)  # Gripper state
        proprio = batch["priv_states"]["articulations"]["panda"][:, 13:31].to(device)
        target_action = batch["action"].to(device)

        # Forward pass
        out = model(z, gripper, proprio)

        # Loss
        loss = F.mse_loss(out, target_action)

        # To have a proper loss printing, we weight this for the size of the batch
        total_loss += loss.item() * z.size(0)

    return total_loss / len(loader.dataset)


# Beware of data leakage: I should split over episodes, not frames themselves
# In a simulation, Frame 45 and Frame 46 of the same episode are 99.9% identical
# If random splitting puts Frame 45 in your Train Set and Frame 46 in your Validation Set, your Validation MSE will drop to near zero
# Train Set: Episodes 0 to 800 (contains all their frames)
# Validation Set: Episodes 800 to 1000 (contains all their frames)
# WARNING: must be the same split of the GNN

# Find the splitting index, as done before
episode_lengths = get_all_episode_lengths(EMBEDDINGS_JS_PATH)
num_train_episodes = int(SPLIT_RATIO * len(episode_lengths))
split_idx = sum(episode_lengths[:num_train_episodes])
print(f"Splitting data for the BC at {split_idx}")

# Create index ranges for Train and Val
train_indices = range(0, split_idx)
val_indices = range(split_idx, len(embeddings_dataset))

# Use Subset to cleanly split the dataset
bc_train_dataset = torch.utils.data.Subset(embeddings_dataset, train_indices)
bc_val_dataset = torch.utils.data.Subset(embeddings_dataset, val_indices)

# Move the data to the dataloaders
bc_train_loader = TorchDataLoader(
    bc_train_dataset, batch_size=BC_BATCH_SIZE, shuffle=True
)
bc_val_loader = TorchDataLoader(bc_val_dataset, batch_size=BC_BATCH_SIZE)

# Prepare the model, optmizer etc.

policy = ResNetGraphStateBCPolicy(
    z_dim=GAT_LATENT_CHANNELS,
    proprio_dim=BC_PROPRIO_DIM,
    gripper_dim=BC_GRIPPER_DIM,
    action_dim=BC_ACTION_DIM,
    hidden_dim=BC_RES_HIDDEN_DIM,
    dropout_p=BC_RES_DROPOUT,
).to(device)

bc_optimizer = torch.optim.AdamW(policy.parameters(), lr=BC_LR)
bc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(bc_optimizer, T_max=BC_EPOCHS)

# Check for existence of a checkpoint and eventually load it
best_bc_val_loss = float("inf")
if BC_CHECKPOINT_PATH.exists():
    bc_checkpoint = torch.load(BC_CHECKPOINT_PATH)
    policy.load_state_dict(bc_checkpoint["model_state_dict"])
    bc_optimizer.load_state_dict(bc_checkpoint["optimizer_state_dict"])
    start_epoch = bc_checkpoint.get("epoch", -1) + 1
    best_bc_val_loss = bc_checkpoint.get("val_loss", float("inf"))
    print(
        f"Loaded BC checkpoint from epoch {start_epoch} with best val loss: {best_bc_val_loss:.5f}"
    )
else:
    # Proceed with training from scratch
    start_epoch = 0
    print("No BC checkpoint found. Starting training from scratch.")

# Train the model for the missing epochs
for epoch in range(start_epoch, BC_EPOCHS):
    train_loss = train_bc_epoch(
        policy,
        bc_train_loader,
        bc_optimizer,
        bc_scheduler,
        device,
    )
    val_loss = validate_bc(policy, bc_val_loader, device)
    print(
        f"BC Epoch {epoch:03d}, Train Action MSE: {train_loss:.5f}, Val Action MSE: {val_loss:.5f}"
    )

    if val_loss < best_bc_val_loss:
        best_bc_val_loss = val_loss
        bc_checkpoint = {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": bc_optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "hyperparameters": {
                "z_dim": GAT_LATENT_CHANNELS,
                "proprio_dim": BC_PROPRIO_DIM,
                "action_dim": BC_ACTION_DIM,
                "hidden_dim": BC_HIDDEN_DIM,
                "batch_size": BC_BATCH_SIZE,
                "lr": BC_LR,
                "epochs": BC_EPOCHS,
            },
        }
        torch.save(bc_checkpoint, BC_CHECKPOINT_PATH)
        print(f"\t>New best BC model saved with Val MSE: {val_loss:.5f}")

# Load the best model weights for later phases
if BC_CHECKPOINT_PATH.exists():
    print(f"Loading best BC model from {BC_CHECKPOINT_PATH}")
    bc_checkpoint = torch.load(BC_CHECKPOINT_PATH)
    policy.load_state_dict(bc_checkpoint["model_state_dict"])

print("\n\n--- Phase 4.1: Baseline Benchmarking ---")


class BaselineBCPolicy(nn.Module):
    def __init__(self, raw_dim, action_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, raw_state):
        return self.mlp(raw_state)


def train_baseline_epoch(model, loader, optimizer, device, noise_std=None):
    model.train()
    total_loss = 0
    for batch in loader:
        # In place of z
        cube = batch["priv_states"]["actors"]["cube"][:, :3].to(device)
        goal = batch["priv_states"]["actors"]["goal_site"][:, :3].to(device)
        table = batch["priv_states"]["actors"]["table-workspace"][:, :3].to(device)
        base = batch["priv_states"]["articulations"]["panda"][:, :3].to(device)

        # As before
        gripper = batch["obs"][:, 18:26].to(device)
        proprio = batch["priv_states"]["articulations"]["panda"][:, 13:31].to(device)
        target_action = batch["action"].to(device)

        state = torch.cat([cube, goal, table, base, gripper, proprio], dim=-1)

        if noise_std is not None:
            state = state + torch.randn_like(state) * noise_std

        optimizer.zero_grad()
        out = model(state)
        loss = F.mse_loss(out, target_action)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * state.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_baseline(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        cube = batch["priv_states"]["actors"]["cube"][:, :3].to(device)
        goal = batch["priv_states"]["actors"]["goal_site"][:, :3].to(device)
        table = batch["priv_states"]["actors"]["table-workspace"][:, :3].to(device)
        base = batch["priv_states"]["articulations"]["panda"][:, :3].to(device)
        gripper = batch["obs"][:, 18:26].to(device)
        proprio = batch["priv_states"]["articulations"]["panda"][:, 13:31].to(device)

        raw_state = torch.cat([cube, goal, table, base, gripper, proprio], dim=-1)
        target_action = batch["action"].to(device)

        out = model(raw_state)
        loss = F.mse_loss(out, target_action)
        total_loss += loss.item() * raw_state.size(0)
    return total_loss / len(loader.dataset)


if BENCHMARK:
    baseline_policy = BaselineBCPolicy(
        raw_dim=38,
        hidden_dim=BC_HIDDEN_DIM,
        action_dim=BC_ACTION_DIM,
    ).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline_policy.parameters(), lr=BC_LR)

    best_baseline_val_loss = float("inf")
    if BASELINE_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(BASELINE_CHECKPOINT_PATH)
        baseline_policy.load_state_dict(checkpoint["model_state_dict"])
        baseline_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_baseline_val_loss = checkpoint.get("val_loss", float("inf"))
        print(
            f"Loaded Baseline checkpoint from epoch {start_epoch} with best val loss: {best_baseline_val_loss:.5f}"
        )
    else:
        start_epoch = 0
        print("No Baseline checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, BC_EPOCHS):
        train_loss = train_baseline_epoch(
            baseline_policy,
            bc_train_loader,
            baseline_optimizer,
            device,
        )
        val_loss = validate_baseline(baseline_policy, bc_val_loader, device)
        if epoch % 1 == 0:
            print(
                f"Baseline Epoch {epoch:03d}, Train MSE: {train_loss:.5f}, Val MSE: {val_loss:.5f}"
            )

        if val_loss < best_baseline_val_loss:
            best_baseline_val_loss = val_loss
            checkpoint = {
                "model_state_dict": baseline_policy.state_dict(),
                "optimizer_state_dict": baseline_optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, BASELINE_CHECKPOINT_PATH)
            print(f"\t>New best Baseline model saved with Val MSE: {val_loss:.5f}")

    if BASELINE_CHECKPOINT_PATH.exists():
        print(f"Loading best Baseline model from {BASELINE_CHECKPOINT_PATH}")
        checkpoint = torch.load(BASELINE_CHECKPOINT_PATH)
        baseline_policy.load_state_dict(checkpoint["model_state_dict"])


print("\n\n--- Phase 4.2: Live Simulator Benchmarking ---")


def evaluate_graph_policy(gae_model, bc_model, num_episodes=100):
    # Reference:
    #   https://gymnasium.farama.org/api/env/
    #   https://gymnasium.farama.org/api/registry/#gymnasium.make
    #   mani_skill/envs/tasks/tabletop/pick_cube.py

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=100,
        render_mode="human" if RENDER else None,
    )

    gae_model.eval()
    bc_model.eval()
    successes = 0

    print(f"Evaluating Graph Policy over {num_episodes} episodes...")
    for seed in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            live_states = env.unwrapped.get_state_dict()

            quick_dataset = [
                {
                    "priv_states": live_states,
                    "obs": obs,
                }
            ]
            built_graph = build_graph(
                quick_dataset, 0
            )  # We only have one frame, so idx=0 is fine

            # Extract node features
            x = built_graph.x.to(device)

            # Extract topology and weights
            edge_index = built_graph.edge_index.to(device)
            edge_attr = built_graph.edge_attr.to(device)

            # Batch array of zeros (all nodes belong to the same single graph)
            batch_idx = torch.zeros(5, dtype=torch.long).to(device)

            with torch.no_grad():
                z = gae_model.encode(x, edge_index, edge_attr, batch_idx)

                gripper = obs[0, 18:26].detach().clone().unsqueeze(0).to(device)
                proprio = (
                    live_states["articulations"]["panda"][0, 13:31]
                    .detach()
                    .clone()
                    .unsqueeze(0)
                    .to(device)
                )

                action = bc_model(z, gripper, proprio)
                action = torch.clamp(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy().squeeze()
            )

            if RENDER:
                env.render()

            if info.get("success", False):
                successes += 1
                break

            done = terminated or truncated

    env.close()
    sr = (successes / num_episodes) * 100
    print(f"Success rate: {sr}%")
    return sr


def evaluate_baseline_policy(baseline_model, num_episodes=100):
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=100,
        render_mode="human" if RENDER else None,
    )

    baseline_model.eval()
    successes = 0

    print(f"Evaluating Baseline Policy over {num_episodes} episodes...")
    for seed in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            live_states = env.unwrapped.get_state_dict()

            cube = live_states["actors"]["cube"][0, :3]
            goal = live_states["actors"]["goal_site"][0, :3]
            table = live_states["actors"]["table-workspace"][0, :3]
            base = live_states["articulations"]["panda"][0, :3]
            gripper = obs[0, 18:26]
            proprio = live_states["articulations"]["panda"][0, 13:31]

            raw_state = np.concatenate([cube, goal, table, base, gripper, proprio])
            raw_state_tensor = (
                torch.tensor(raw_state, dtype=torch.float).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                action = baseline_model(raw_state_tensor)
                action = torch.clamp(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy().squeeze()
            )

            if RENDER:
                env.render()

            if info.get("success", False):
                successes += 1
                break

            done = terminated or truncated

    env.close()
    sr = (successes / num_episodes) * 100
    print(f"Success rate: {sr}%")
    return sr


graph_sr = evaluate_graph_policy(model, policy)
if BENCHMARK:
    baseline_sr = evaluate_baseline_policy(baseline_policy)
