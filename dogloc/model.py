import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _conv_bn(ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(ch),
        nn.SiLU(),
    )


class BiFPNBlock(nn.Module):
    """One BiFPN layer over three feature levels with fast normalised fusion."""

    def __init__(self, ch: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w_td4 = nn.Parameter(torch.ones(2))
        self.w_td3 = nn.Parameter(torch.ones(2))
        self.w_bu4 = nn.Parameter(torch.ones(3))
        self.w_bu5 = nn.Parameter(torch.ones(2))
        self.conv_td4, self.conv_td3 = _conv_bn(ch), _conv_bn(ch)
        self.conv_bu4, self.conv_bu5 = _conv_bn(ch), _conv_bn(ch)

    def _fuse(self, w: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(w)
        w = w / (w.sum() + self.eps)
        return sum(wi * f for wi, f in zip(w, feats))

    def forward(self, p3, p4, p5):
        p4_td = self.conv_td4(self._fuse(self.w_td4, [p4, F.interpolate(p5, size=p4.shape[-2:])]))
        p3_out = self.conv_td3(self._fuse(self.w_td3, [p3, F.interpolate(p4_td, size=p3.shape[-2:])]))
        p4_out = self.conv_bu4(self._fuse(self.w_bu4, [p4, p4_td, F.max_pool2d(p3_out, 2)]))
        p5_out = self.conv_bu5(self._fuse(self.w_bu5, [p5, F.max_pool2d(p4_out, 2)]))
        return p3_out, p4_out, p5_out


class DogLocalizer(nn.Module):
    """EfficientNet-B0 backbone + BiFPN neck, with a classification head and a single-box head.

    This is a single-object localizer (one box per image), not an anchor-based detector.
    The box head keeps spatial layout from the stride-8 map, which global pooling would lose.
    """

    def __init__(self, num_classes: int, ch: int = 64, bifpn_repeats: int = 3,
                 pretrained: bool = True, cls_dropout: float = 0.3):
        super().__init__()
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        features = tvm.efficientnet_b0(weights=weights).features
        self.backbone = nn.ModuleDict({
            "c3": features[:4],    # 40 ch, stride 8
            "c4": features[4:6],   # 112 ch, stride 16
            "c5": features[6:9],   # 1280 ch, stride 32
        })
        self.lateral = nn.ModuleDict({
            "c3": nn.Conv2d(40, ch, 1, bias=False),
            "c4": nn.Conv2d(112, ch, 1, bias=False),
            "c5": nn.Conv2d(1280, ch, 1, bias=False),
        })
        self.bifpn = nn.ModuleList([BiFPNBlock(ch) for _ in range(bifpn_repeats)])

        self.cls_head = nn.Sequential(
            nn.Linear(ch * 3, 256), nn.SiLU(), nn.Dropout(cls_dropout),
            nn.Linear(256, num_classes),
        )
        self.box_spatial = nn.Sequential(
            *[_conv_bn(ch, stride=2) for _ in range(4)],
            # Keeps the head input size fixed for any input resolution.
            nn.AdaptiveAvgPool2d(4),
        )
        self.box_head = nn.Sequential(
            nn.Linear(ch * 16 + ch * 2, 256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor):
        c3 = self.backbone["c3"](x)
        c4 = self.backbone["c4"](c3)
        c5 = self.backbone["c5"](c4)
        p3, p4, p5 = self.lateral["c3"](c3), self.lateral["c4"](c4), self.lateral["c5"](c5)
        for block in self.bifpn:
            p3, p4, p5 = block(p3, p4, p5)

        g3, g4, g5 = (F.adaptive_avg_pool2d(p, 1).flatten(1) for p in (p3, p4, p5))
        logits = self.cls_head(torch.cat([g3, g4, g5], 1))

        raw = self.box_head(torch.cat([self.box_spatial(p3).flatten(1), g4, g5], 1))
        cxcy, wh = raw[:, :2].sigmoid(), raw[:, 2:].sigmoid()
        box = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1).clamp(0, 1)
        return logits, box

    def param_groups(self, lr_backbone: float, lr_head: float, lr_box: float) -> list[dict]:
        box_params = list(self.box_spatial.parameters()) + list(self.box_head.parameters())
        backbone_params = list(self.backbone.parameters())
        taken = {id(p) for p in box_params + backbone_params}
        other = [p for p in self.parameters() if id(p) not in taken]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": other, "lr": lr_head},
            {"params": box_params, "lr": lr_box},
        ]
