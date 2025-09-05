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
    """Setup observations from [progress, compass, depth] tensor list."""
    progress_tensor, compass_tensor, depth_tensor = obs_list

    N, A = progress_tensor.shape[0:2]
    batch_size = N * A

    # Reshape all tensors to batch format
    obs_tensors = [
        progress_tensor.view(batch_size, *progress_tensor.shape[2:]),  # [batch, 1]
        compass_tensor.view(batch_size, *compass_tensor.shape[2:]),  # [batch, 128]
        depth_tensor.view(batch_size, *depth_tensor.shape[2:]),  # [batch, 128, 1]
    ]

    # Calculate total features: 1 (progress) + 128 (compass) + 128 (depth flattened)
    num_obs_features = 1 + 128 + 128  # = 257

    return obs_tensors, num_obs_features


def process_obs(progress, compass, depth):
    assert not torch.isnan(progress).any()
    assert not torch.isinf(progress).any()

    assert not torch.isnan(compass).any()
    assert not torch.isinf(compass).any()

    assert not torch.isnan(depth).any()
    assert not torch.isinf(depth).any()

    return torch.cat(
        [
            progress.view(progress.shape[0], -1),  # [batch, 1]
            compass.view(compass.shape[0], -1),  # [batch, 128]
            depth.view(depth.shape[0], -1),  # [batch, 128]
        ],
        dim=1,
    )


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
