import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ResEncoderPaper(nn.Module):
    def __init__(self):
        super(ResEncoderPaper, self).__init__()
        self.model = resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.repr_dim = 1024
        self.image_channel = 3
        x = torch.randn([64] + [3, 224, 224])
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        #
        # Initialization
        nn.init.orthogonal_(self.fc.weight.data)
        self.fc.bias.data.fill_(0.0)

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        # time_step = obs.shape[1] // self.image_channel
        # obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        # obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == 'layer2':
                break

        # conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        # conv_current = conv[:, 1:, :, :, :]
        # conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        # conv = torch.cat([conv_current, conv_prev], axis=1)
        # conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        # if flatten:
        #     conv = conv.view(conv.size(0), -1)

        # flatten the output
        obs = obs.view(obs.size(0), -1)

        return obs


    def forward(self, obs):
        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)
        # obs = self.model(self.transform(obs.to(torch.float32)) / 255.0 - 0.5)
        return out
    
class ResEncoderSimple(nn.Module):
    def __init__(self):
        super(ResEncoderSimple, self).__init__()
        self.encoder = resnet18(pretrained=True)

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.repr_dim = 1024

        x = torch.randn([64] + [3, 224, 224])
        with torch.no_grad():
            out_shape = self.encoder(x).shape
        self.out_dim = out_shape[1]

        self.fc = nn.Linear(self.out_dim, self.repr_dim)

        self.transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    
    def forward(self, obs):
        # obs = obs / 255.0 - 0.5
        obs = self.transform(obs.to(torch.float32))    
        out = self.encoder(obs)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class ResnetCNN(BaseFeaturesExtractor):

    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing the observations
        self.cnn = ResEncoderSimple()
        self.n_input_channels = observation_space.shape[0]
        assert self.cnn.repr_dim == features_dim, f"Expected features_dim to be {self.cnn.repr_dim}, got {features_dim}"

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # if n_input_channels == 1 stack the images to have 3 input channels
        if self.n_input_channels == 1:
            observations = torch.cat((observations, observations, observations), dim=1)
        
        return self.cnn(observations)
    

if __name__ == '__main__':
    
    from stable_baselines3 import SAC
    from packingGame import PackingGame

    policy_kwargs = dict(
        features_extractor_class=ResnetCNN,
        features_extractor_kwargs=dict(features_dim=1024),
    )

    env = PackingGame(visual=False, ordered_objs=True, reward_function='compactness_stability')

    model = SAC('CnnPolicy', env=env, policy_kwargs=policy_kwargs, verbose=1)

    model.learn(1000)