# import copy
# import torch
# import torch.nn as nn
# from typing import Optional, Tuple


# class MLPTanhHead(torch.nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super().__init__()
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, 1024),
#             torch.nn.ReLU(),
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, output_size),
#             torch.nn.Tanh(),
#         )

#     def forward(self, x):
#         return self.mlp(x)


# class DeterministicDecoder(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         window_size: int,
#         out_features: int = 7,
#         hidden_size: int = 4096,
#         multi_step_action=1,
#     ):
#         super(DeterministicDecoder, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.window_size = window_size
#         self.multi_step_action = multi_step_action
#         self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
#         self.hidden_state = None
#         self.hidden_size = hidden_size
#         self.global_1d_pool = nn.AdaptiveMaxPool1d(1)


#     def forward(  # type: ignore
#         self,
#         input_feature: torch.Tensor,
#         h_0: Optional[torch.Tensor] = None,
#     ):
        
#         # reshape
#         if input_feature.dim() == 3:
#             input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
#         input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        
#         actions = self.actions(input_feature)

#         return actions



# lm_head = DeterministicDecoder(4096, 1, multi_step_action=1)


# output_hs = torch.rand(1, 631, 4096)
# output_hs = lm_head(output_hs)
# print('1')


# v2



import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple



class ActionDecoder(nn.Module):
    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_and_act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def clear_hidden_state(self) -> None:
        pass

class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class DeterministicDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        out_features: int = 76,
        hidden_size: int = 4096,
        multi_step_action=1,
    ):
        super(DeterministicDecoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)

        

    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def forward(  # type: ignore
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ):
        
        # reshape
        if input_feature.dim() == 3:
            input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        
        actions = self.actions(input_feature)

        return actions


    def act(
        self,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions, self.hidden_state = self(
            input_feature, self.hidden_state
        )

        return pred_actions


