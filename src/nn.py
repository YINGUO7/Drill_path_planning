import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CoverageCNN(nn.Module):
    """覆盖栅格特征提取器。

    输入：
        coverage: Tensor，形状为(B, 1, H, W)，类别值为0到3。

    输出：
        feature: Tensor，形状为(B, out_dim)。
    """

    def __init__(self, input_shape, out_dim=128):
        super().__init__()
        channels, height, width = input_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flat_dim = int(self.cnn(dummy).flatten(1).shape[1])
        self.head = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.cnn(x).flatten(1))


class MillingNurbsFeatureExtractor(BaseFeaturesExtractor):
    """Env7的SAC多输入特征提取器。

    输入：
        observation_space: Dict空间，包含：
            state: 低维标量状态。
            coverage: 单通道覆盖类别图。
        features_dim: 输出特征维度。

    输出：
        fused_feature: SAC策略网络和Q网络共用的特征向量。
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        state_dim = int(observation_space["state"].shape[0])
        coverage_shape = observation_space["coverage"].shape

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.coverage_cnn = CoverageCNN(coverage_shape, out_dim=128)

        self.fusion = nn.Sequential(
            nn.Linear(64 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
        )
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        state = obs["state"].float()
        coverage = obs["coverage"].float() / 3.0
        state_feature = self.state_mlp(state)
        coverage_feature = self.coverage_cnn(coverage)
        x = self.fusion(torch.cat((state_feature, coverage_feature), dim=1))
        x = x + self.residual(x)
        return self.output(x)
