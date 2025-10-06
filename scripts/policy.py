import math

import torch
from madrona_escape_room_learn import (
    ActorCritic,
    BackboneEncoder,
    BackboneSeparate,
    BackboneShared,
    RecurrentBackboneEncoder,
)
from madrona_escape_room_learn.models import (
    MLP,
    LinearLayerCritic,
    LinearLayerDiscreteActor,
)
from madrona_escape_room_learn.rnn import LSTM


def setup_obs(obs_list):
    """Setup observations from tensor list. Supports compass+lidar setup."""
    if len(obs_list) == 2:
        # Current setup: [compass, lidar]
        compass_tensor, lidar_tensor = obs_list

        N, A = compass_tensor.shape[0:2]
        batch_size = N * A

        # Get actual feature sizes from tensor shapes
        compass_features = compass_tensor.shape[2]  # Dynamically get compass size (e.g., 256)
        lidar_features = lidar_tensor.shape[2]  # Dynamically get lidar size (e.g., 256)

        # Reshape tensors to batch format
        obs_tensors = [
            compass_tensor.view(batch_size, compass_features),
            lidar_tensor.view(batch_size, lidar_features),
        ]

        # Calculate total features from actual tensor shapes
        num_obs_features = compass_features + lidar_features

        return obs_tensors, num_obs_features

    else:
        raise ValueError(f"Expected 2 observation tensors (compass + lidar), got {len(obs_list)}")


def process_obs(*obs_tensors):
    """Process observation tensors. Handles both minimal and full observation setups."""

    # Validate all tensors
    for tensor in obs_tensors:
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()

    # Flatten and concatenate all tensors
    flattened_tensors = [tensor.view(tensor.shape[0], -1) for tensor in obs_tensors]

    return torch.cat(flattened_tensors, dim=1)


def make_policy(num_obs_features, num_channels, separate_value):
    # encoder = RecurrentBackboneEncoder(
    #    net = MLP(
    #        input_dim = num_obs_features,
    #        num_channels = num_channels,
    #        num_layers = 2,
    #    ),
    #    rnn = LSTM(
    #        in_channels = num_channels,
    #        hidden_channels = num_channels,
    #        num_layers = 1,
    #    ),
    # )

    encoder = BackboneEncoder(
        net=MLP(
            input_dim=num_obs_features,
            num_channels=num_channels,
            num_layers=3,
        ),
    )

    if separate_value:
        backbone = BackboneSeparate(
            process_obs=process_obs,
            actor_encoder=encoder,
            critic_encoder=RecurrentBackboneEncoder(
                net=MLP(
                    input_dim=num_obs_features,
                    num_channels=num_channels,
                    num_layers=2,
                ),
                rnn=LSTM(
                    in_channels=num_channels,
                    hidden_channels=num_channels,
                    num_layers=1,
                ),
            ),
        )
    else:
        backbone = BackboneShared(
            process_obs=process_obs,
            encoder=encoder,
        )

    return ActorCritic(
        backbone=backbone,
        actor=LinearLayerDiscreteActor(
            [4, 8, 5],
            num_channels,
        ),
        critic=LinearLayerCritic(num_channels),
    )
