# -*- coding: utf-8 -*-
# type: ignore
"""# Resources:

*   https://github.com/haosulab/ManiSkill/tree/main
*   https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
*   http://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html
"""

import random
import argparse
from typing import Union
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import math
from collections import deque

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader


from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common
import gymnasium as gym

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from gnn import (
    print_dict_tree,
    get_all_episode_lengths,
    build_graph,
    GATAutoencoder,
    NUM_NODES,
    IN_CHANNELS,
    GAT_CHECKPOINT_PATH,
    GAT_HIDDEN_CHANNELS,
    GAT_LATENT_CHANNELS,
    GAT_ATTENTION_HEADS,
)


# Argument Parsing
parser = argparse.ArgumentParser(
    description="ManiSkill GAT-BC Training and Benchmarking"
)
parser.add_argument(
    "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
)
parser.add_argument(
    "--render",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable rendering",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="diff_unet_policy_best.pth",
    help="Diffusion checkpoint filename (default: diff_unet_policy_best.pth)",
)

args = parser.parse_args()


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
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
CHECKPOINT_PATH = script_location / args.checkpoint

# Flags for run later
RENDER = args.render

# Generic Hyperparameters
SPLIT_RATIO = 0.8

# BC Hyperparameters
EPOCHS = args.epochs
LR = 1e-4
BATCH_SIZE = 128
ACTION_DIM = 4
PROPRIO_DIM = 18
GRIPPER_DIM = 8
Z_DIM = GAT_LATENT_CHANNELS * NUM_NODES
UNET_DIMS = [64, 128, 256]
UNET_GROUPS = 8

# Diffusion Hyperparameters
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_HORIZON = 8
NUM_DIFFUSION_ITERS = 100
DIFF_STEP_EMBED_DIM = 64

# *** NOTES ON HOW TO ACCESS DATA ***
#
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
#
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
# ** Comments **
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


# In ManiSkill examples/, the PickCube-v1 task is addressed using three primary architectures:
#
#   1. Behavioral Cloning (BC): A MLP with two hidden layers of 256 units and ReLU activations
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
#   Reference
#       examples/baselines/bc,
#       examples/baselines/act,
#       examples/baselines/diffusion_policy


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
    def __init__(
        self,
        dataset_file: str,
        obs_horizon: int,
        pred_horizon: int,
        load_count=-1,
        success_only: bool = False,
        device=None,
    ) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        self.data = h5py.File(dataset_file, "r")
        json_path = Path(dataset_file).with_suffix(".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        # Instead of flattening, we keep a LIST of episodes.
        # Each element is a dict containing the full episode's obs, actions, and priv_states.
        self.trajectories = []

        if load_count == -1:
            load_count = len(self.episodes)

        print(f"Loading {load_count} episodes into memory for sequence extraction...")
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only and not eps.get("success", False):
                continue

            traj_data = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(traj_data)
            eps_len = len(trajectory["actions"])

            # Extract the raw dict arrays for this specific episode
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            priv_states = common.index_dict_array(
                trajectory["env_states"], slice(eps_len)
            )
            actions = trajectory["actions"]

            # Store them securely
            self.trajectories.append(
                {"obs": obs, "priv_states": priv_states, "actions": actions}
            )

        # Pre-compute sliding windows
        # We calculate every valid (traj_index, start_frame, end_frame, full_traj_lenght) tuple
        # |o|o|                             observations: 2 (obs_horizon)
        # | |a|a|a|a|a|a|a|a|               actions executed: 8 (actions_horizon)
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16 (pred_horizon)
        #    ^
        # You are here! (You only have this one)
        # So we need 1 frames before (and 14 after), indeed, 1 + 1 + 14 = 16
        self.slices = []
        pad_before = (
            self.obs_horizon - 1
        )  # Number of input frames we need before the current frame (i.e., the -1 in the formula) to fill the observation horizon

        # We need the end to stretch far enough to get all target actions
        # but the actual actions we take are bounded by the prediction horizon
        for traj_idx, traj in enumerate(self.trajectories):
            L = len(traj["actions"])

            # |o|o|                             observations: 2 (obs_horizon)
            # | |a|a|a|a|a|a|a|a|               actions executed: 8 (actions_horizon)
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16 (pred_horizon)
            #    ^
            # You are here! (You only have this one)
            # So we need 14 frames after
            pad_after = self.pred_horizon - self.obs_horizon

            # Generate windows that slide across the entire trajectory (e.g. for traj_0: from -1 to 74-16+14=72)
            for start in range(-pad_before, L - self.pred_horizon + pad_after):
                end = start + self.pred_horizon
                self.slices.append((traj_idx, start, end, L))

        print(
            f"Dataset initialized: {len(self.slices)} valid temporal windows extracted."
        )

    def _slice_and_pad(self, data, start, end, L):
        """
        Recursively slices and pads dictionaries or numpy arrays.
        - If start < 0, it repeats the first frame.
        - If end > L, it repeats the last frame.
        """
        if isinstance(data, dict):
            # If it's a nested dictionary (like priv_states), recurse!
            return {k: self._slice_and_pad(v, start, end, L) for k, v in data.items()}

        # Base case: It's a numpy array
        pad_before = max(0, -start)
        pad_after = max(0, end - L)

        valid_start = max(0, start)
        valid_end = min(L, end)

        # Grab the valid segment
        seq = data[valid_start:valid_end]

        # Pad by repeating the first frame
        if pad_before > 0:
            padding = np.repeat(seq[0:1], pad_before, axis=0)
            seq = np.concatenate([padding, seq], axis=0)

        # Pad by repeating the last frame
        if pad_after > 0:
            padding = np.repeat(seq[-1:], pad_after, axis=0)
            seq = np.concatenate([seq, padding], axis=0)

        return seq

    def __len__(self):
        # The length is now the number of windows, not the number of frames
        return len(self.slices)

    def __getitem__(self, idx):
        # Identify which window we are pulling
        traj_idx, start, end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        # Slice and pad observations and priv_states (need obs_horizon length only)
        obs_seq = self._slice_and_pad(traj["obs"], start, start + self.obs_horizon, L)
        priv_states_seq = self._slice_and_pad(
            traj["priv_states"], start, start + self.obs_horizon, L
        )

        # Slice and pad actions (needs the full pred_horizon length)
        action_seq = self._slice_and_pad(traj["actions"], start, end, L)

        # Convert to tensors (common.to_tensor recursively handles dictionaries natively)
        if self.device is not None:
            obs_seq = common.to_tensor(obs_seq, device=self.device)
            priv_states_seq = common.to_tensor(priv_states_seq, device=self.device)
            action_seq = common.to_tensor(action_seq, device=self.device)

        return {
            "obs_seq": obs_seq,
            "priv_states_seq": priv_states_seq,
            "action_seq": action_seq,
        }


dataset = ManiSkillTrajectoryDataset(EMBEDDINGS_H5_PATH, OBS_HORIZON, PRED_HORIZON)
print(
    f"""Dataset lenght: {
        len(dataset)
    }, i.e., number of windows extracted (-> (traj where the windows was extracted, start frame in such trajectory, end fram in such trajectory, full lenght of the trajectory) 4-ple for training later"""
)
print_dict_tree(dataset[0])


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        #  time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class DiffusionAgent(nn.Module):
    def __init__(
        self,
        obs_horizon,
        act_horizon,
        pred_horizon,
        action_dim,
        obs_dim,
        diffusion_step_embed_dim,
        unet_dims,
        n_groups,
        num_diffusion_iters=100,
    ):

        super().__init__()
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = action_dim
        self.obs_dim = obs_dim

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care) NOTE: I dont agree, anyway, this is like RGB for images
            global_cond_dim=self.obs_horizon * self.obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=obs_seq.device,
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq.device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


def train_epoch(model, loader, optimizer, scheduler, ema, device):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()  # Best practice is to zero_grad at the start of the loop

        # Extract data, keeping the time dimension intact -> [Batch, Obs_Horizon, ...]
        z = batch["priv_states_seq"]["embeddings"].to(device)
        gripper = batch["obs_seq"][:, :, 18:26].to(
            device
        )  # Notice the extra ':' for the time dimension!
        proprio = batch["priv_states_seq"]["articulations"]["panda"][:, :, 13:31].to(
            device
        )

        target_action = batch["action_seq"].to(device)  # [Batch, Pred_Horizon, 4]

        # Ensure z is flattened across the node dimension -> [Batch, Obs_Horizon, 40]
        B, L = gripper.shape[0], gripper.shape[1]
        z = z.view(B, L, -1)

        # Concatenate into a single conditioning sequence -> [Batch, Obs_Horizon, 66]
        obs_seq = torch.cat([z, gripper, proprio], dim=-1)

        # Forward pass
        loss = model.compute_loss(
            obs_seq=obs_seq,
            action_seq=target_action,
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # In Diffusion, LR schedulers step every batch
        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.step(model.parameters())

        # Weight the loss by batch size for an accurate epoch average
        total_loss += loss.item() * B

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0

    for batch in loader:
        # Extract data
        z = batch["priv_states_seq"]["embeddings"].to(device)
        gripper = batch["obs_seq"][:, :, 18:26].to(device)
        proprio = batch["priv_states_seq"]["articulations"]["panda"][:, :, 13:31].to(
            device
        )

        target_action = batch["action_seq"].to(device)

        B, L = gripper.shape[0], gripper.shape[1]
        z = z.view(B, L, -1)

        # Concatenate
        obs_seq = torch.cat([z, gripper, proprio], dim=-1)

        # Calculate validation loss
        # we use compute_loss here instead of full denoising
        # running 100 diffusion steps for every validation batch would take hours
        loss = model.compute_loss(
            obs_seq=obs_seq,
            action_seq=target_action,
        )

        total_loss += loss.item() * B

    return total_loss / len(loader.dataset)


# Split the dataset
# NOTE: must be the same split of the GNN
episode_lengths = get_all_episode_lengths(EMBEDDINGS_JS_PATH)
num_train_episodes = int(SPLIT_RATIO * len(episode_lengths))
print(
    f"Splitting dataset: {num_train_episodes} Train Episodes, {len(episode_lengths) - num_train_episodes} Val Episodes"
)

train_indices = []
val_indices = []

# dataset.slices contains (traj_idx, start, end, L)
for i, slice_tuple in enumerate(dataset.slices):
    traj_idx = slice_tuple[0]
    if traj_idx < num_train_episodes:
        train_indices.append(i)
    else:
        val_indices.append(i)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Move the data to the dataloaders
train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = TorchDataLoader(val_dataset, batch_size=BATCH_SIZE)

# NOTE: Should really move on and forget about EPOCHS as people in this field do
total_training_steps = len(train_loader) * EPOCHS

# Prepare the model, optmizer, eventually scheduler etc.
policy = DiffusionAgent(
    obs_horizon=OBS_HORIZON,
    act_horizon=ACT_HORIZON,
    pred_horizon=PRED_HORIZON,
    action_dim=ACTION_DIM,
    obs_dim=Z_DIM + PROPRIO_DIM + GRIPPER_DIM,
    diffusion_step_embed_dim=DIFF_STEP_EMBED_DIM,
    unet_dims=UNET_DIMS,
    n_groups=UNET_GROUPS,
).to(device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
lr_scheduler = get_scheduler(  # just as ManiSkill does
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=total_training_steps,
)
ema = EMAModel(
    policy.parameters(),
    decay=0.999,  # 0.999 is standard for Diffusion
    inv_gamma=1.0,
    power=0.75,
)

# Check for existence of a checkpoint and eventually load it
best_val_loss = float("inf")
if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH)
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("val_loss", float("inf"))
    print(
        f"Loaded Diffusion checkpoint from epoch {start_epoch} with best val loss: {best_val_loss:.5f}"
    )
else:
    # Proceed with training from scratch
    start_epoch = 0
    print("No Diffusion checkpoint found. Starting training from scratch.")

# Train the model for the missing epochs
for epoch in range(start_epoch, EPOCHS):
    train_loss = train_epoch(
        policy,
        train_loader,
        optimizer,
        lr_scheduler,
        ema,
        device,
    )
    val_loss = validate(policy, val_loader, device)
    print(
        f"Diffusion epoch {epoch:03d}, Train Action MSE: {train_loss:.5f}, Val Action MSE: {val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            "model_state_dict": policy.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "hyperparameters": {
                "obs_horizon": OBS_HORIZON,
                "act_horizon": ACT_HORIZON,
                "pred_horizon": PRED_HORIZON,
                "action_dim": ACTION_DIM,
                "obs_dim": Z_DIM + PROPRIO_DIM + GRIPPER_DIM,
                "diffusion_step_embed_dim": DIFF_STEP_EMBED_DIM,
                "unet_dims": UNET_DIMS,
                "n_groups": UNET_GROUPS,
                "num_diffusion_iters": NUM_DIFFUSION_ITERS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "split_ratio": SPLIT_RATIO,
            },
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"\t>New best diffusion model saved with Val MSE: {val_loss:.5f}")

# Load the best model weights
if CHECKPOINT_PATH.exists():
    print(f"Loading best diffusion model from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    policy.load_state_dict(checkpoint["model_state_dict"])


def evaluate_graph_policy(gae_model, diff_model, num_episodes=100):
    # Reference:
    #   https://gymnasium.farama.org/api/env/
    #   mani_skill/envs/tasks/tabletop/pick_cube.py

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=100,
        render_mode="human" if RENDER else None,
    )

    gae_model.eval()
    diff_model.eval()
    successes = 0

    print(f"Evaluating Diffusion Graph Policy over {num_episodes} episodes...")
    for seed in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=seed)
        done = False

        # Initialize the rolling buffer for the observation history
        obs_buffer = deque(maxlen=OBS_HORIZON)

        # Helper function to extract the 66D vector from the current environment state
        def get_current_obs_vector(current_obs, current_env):
            live_states = current_env.unwrapped.get_state_dict()

            # Build the graph and encode it on the fly
            quick_dataset = [{"priv_states": live_states, "obs": current_obs}]
            built_graph = build_graph(quick_dataset, 0).to(device)

            with torch.no_grad():
                # gae_model.encode returns [1, 40]. Squeeze it to [40] for the buffer
                z = gae_model.encode(built_graph).squeeze(0)

            # Extract gripper and proprio
            gripper = current_obs[0, 18:26].detach().clone().to(device)
            proprio = (
                live_states["articulations"]["panda"][0, 13:31]
                .detach()
                .clone()
                .to(device)
            )

            # Concatenate into the final 66D vector
            return torch.cat([z.flatten(), gripper, proprio], dim=-1)

        # Bootstrap the buffer with the first observation (pad the beginning)
        first_obs_vec = get_current_obs_vector(obs, env)
        for _ in range(OBS_HORIZON):
            obs_buffer.append(first_obs_vec)

        step_count = 0
        while not done and step_count < 100:
            # Stack the rolling buffer into a sequence: [1, OBS_HORIZON, 66]
            obs_seq = torch.stack(list(obs_buffer)).unsqueeze(0)

            # Diffusion process: Denoise 100 times to get the action sequence
            with torch.no_grad():
                # Returns [1, ACT_HORIZON, ACTION_DIM]
                action_chunk = diff_model.get_action(obs_seq)
                action_chunk = action_chunk.squeeze(0).cpu().numpy()

            # Inner Execution Loop: Execute the predicted chunk up to ACT_HORIZON
            for i in range(ACT_HORIZON):
                action = action_chunk[i]
                action = np.clip(action, -1.0, 1.0)  # Always clamp real robot actions!

                # Step the simulator
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                if RENDER:
                    env.render()

                if info.get("success", False):
                    successes += 1
                    done = True
                    break

                done = terminated or truncated
                if done:
                    break

                # Update the rolling buffer with the new physical observation, this automatically pushes the oldest frame out!
                new_obs_vec = get_current_obs_vector(obs, env)
                obs_buffer.append(new_obs_vec)

    env.close()
    sr = (successes / num_episodes) * 100
    print(f"Success rate: {sr}%")
    return sr


ema.copy_to(policy.parameters())
policy.eval()

gnn_model = GATAutoencoder(
    NUM_NODES,
    IN_CHANNELS,
    GAT_HIDDEN_CHANNELS,
    GAT_LATENT_CHANNELS,
    GAT_ATTENTION_HEADS,
).to(device)
if GAT_CHECKPOINT_PATH.exists():
    print(f"Loading best GAT model from {GAT_CHECKPOINT_PATH}")
    checkpoint = torch.load(GAT_CHECKPOINT_PATH)
    gnn_model.load_state_dict(checkpoint["model_state_dict"])

graph_sr = evaluate_graph_policy(gnn_model, policy)
