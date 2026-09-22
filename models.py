import torch.nn as nn


def conv_block(in_ch, out_ch):
    """Two conv layers + BatchNorm + ReLU, then halve the image size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 64x64 -> 32x32 -> 16x16 -> 8x8, learning more detailed patterns at each stage
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            # average each of the 128 patterns into one number
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # randomly switch off 30% during training to avoid memorising
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # one raw score per sign type
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_classes, shape_color=False):
    if shape_color:
        raise NotImplementedError(
            "Shape and colour branch (S) is not built yet")
    return TrafficSignCNN(num_classes)
