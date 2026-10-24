import torch
import torch.nn as nn


# =========================================================== LEGACY (unchanged)
class oneD_DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.LeakyReLU(0.2), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + res)


class oneD_CNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
                                  nn.BatchNorm1d(32), nn.LeakyReLU(0.2))
        self.layer1 = nn.Sequential(ResidualBlock1D(32), nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64),
                                    nn.LeakyReLU(0.2), ResidualBlock1D(64), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        x = self.layer2(self.layer1(self.stem(x))).squeeze(-1)
        return self.head(x)


# =========================================================== SPEC (Bangaram PDF)
def make_norm(kind, channels, groups=8):
    if kind == "group":
        g = min(groups, channels)
        assert channels % g == 0, (channels, g)
        return nn.GroupNorm(g, channels)
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class SpecDAE(nn.Module):
    """PDF: 216x1 -> Conv32 -> Pool -> Conv16 -> Pool -> Conv16 -> Up -> Conv32 -> Up -> Conv1.
    Kernel size and activation are NOT in the PDF. ASSUMED: k=3, ReLU, linear output."""

    def __init__(self, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, padding=p), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size, padding=p), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, kernel_size, padding=p), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 32, kernel_size, padding=p), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 1, kernel_size, padding=p),
        )

    def forward(self, x):
        return self.net(x)


class SpecCNN(nn.Module):
    """PDF: Conv32(k5)-Norm-Pool -> Conv64(k5)-Norm-Pool -> Conv128(k3) -> GAP -> Dense64 -> Drop -> Dense1.
    Activation is NOT in the PDF. ASSUMED: ReLU, 'same' padding. Returns LOGITS."""

    def __init__(self, norm="group", gn_groups=8, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), make_norm(norm, 32, gn_groups), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), make_norm(norm, 64, gn_groups), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=-1))


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)