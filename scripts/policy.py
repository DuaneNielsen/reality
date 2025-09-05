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
    """Setup observations from tensor list. Supports minimal and lidar setups."""
    if len(obs_list) == 2:
        # Minimal setup: [progress, compass]
        progress_tensor, compass_tensor = obs_list

        N, A = progress_tensor.shape[0:2]
        batch_size = N * A

        # Reshape tensors to batch format
        obs_tensors = [
            progress_tensor.view(batch_size, *progress_tensor.shape[2:]),  # [batch, 1]
            compass_tensor.view(batch_size, *compass_tensor.shape[2:]),  # [batch, 128]
        ]

        # Calculate total features: 1 (progress) + 128 (compass)
        num_obs_features = 1 + 128  # = 129

        return obs_tensors, num_obs_features

    elif len(obs_list) == 3:
        # Full setup: [progress, compass, lidar/depth]
        progress_tensor, compass_tensor, sensor_tensor = obs_list

        N, A = progress_tensor.shape[0:2]
        batch_size = N * A

        # Sensor tensor is either lidar depth [worlds, agents, 128, 1] or depth image
        # Both need to be flattened to [batch, features]
        sensor_reshaped = sensor_tensor.view(batch_size, -1)
        sensor_features = sensor_reshaped.shape[-1]

        # Reshape all tensors to batch format
        obs_tensors = [
            progress_tensor.view(batch_size, *progress_tensor.shape[2:]),  # [batch, 1]
            compass_tensor.view(batch_size, *compass_tensor.shape[2:]),  # [batch, 128]
            sensor_reshaped,  # [batch, sensor_features]
        ]

        # Calculate total features
        num_obs_features = 1 + 128 + sensor_features

        return obs_tensors, num_obs_features

    else:
        raise ValueError(f"Expected 2 or 3 observation tensors, got {len(obs_list)}")


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
