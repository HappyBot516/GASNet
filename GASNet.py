import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from backbone import build_backbone


class GASNet(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet101', pretrained=True):
        super(GASNet, self).__init__()
        self.num_classes = num_classes

        # Adaptive ConvNeXt backbone
        self.adaptive_backbone = build_backbone(backbone_type)
        self.convnext_context = models.convnext_base(pretrained=pretrained)
        for param in self.convnext_context.parameters():
            param.requires_grad = False

        # Feature pyramid lateral connections
        self.top_proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.lat_conv4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lat_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat_conv2 = nn.Conv2d(256, 256, kernel_size=1)

        # Decoder (Multi-Scale Feature Refiner)
        self.refiner4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refiner3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refiner2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Progressive resolution-aware supervision
        self.semantic_refine = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.supervise_s2 = nn.Conv2d(256, 1, kernel_size=1)
        self.supervise_s3 = nn.Conv2d(256, 1, kernel_size=1)
        self.supervise_s4 = nn.Conv2d(256, 1, kernel_size=1)
        self.supervise_s5 = nn.Conv2d(256, 1, kernel_size=1)

        # Dilated convolutions for context
        self.context_dil2 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.context_dil4 = nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4)
        self.context_dil6 = nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6)
        self.context_dil8 = nn.Conv2d(256, 256, kernel_size=3, padding=8, dilation=8)

        # Final prediction head
        self.class_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier = nn.Linear(128, num_classes)

        # GroupNorm
        self.norm_semantic = nn.GroupNorm(128, 128)
        self.norm_context = nn.GroupNorm(256, 256)

    def _resize_to(self, x, target_h, target_w):
        return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)

    def _scale_up(self, x, factor):
        _, _, h, w = x.shape
        return F.interpolate(x, size=(h * factor, w * factor), mode='bilinear', align_corners=True)

    def _scale_down(self, x, factor):
        _, _, h, w = x.shape
        return F.interpolate(x, size=(h // factor, w // factor), mode='bilinear', align_corners=True)

    def _fuse(self, low_feat, high_feat):
        _, _, H, W = high_feat.shape
        return F.interpolate(low_feat, size=(H, W), mode='bilinear', align_corners=True) + high_feat

    def forward(self, x, gabor_mask):
        x = self._scale_up(x, 4)

        # Adaptive ConvNeXt backbone
        features = self.adaptive_backbone(x)
        c1, c2, c3, c4, c5 = features

        # Top-down feature pathway
        p5 = self.top_proj(c5)
        p4 = self._fuse(p5, self.lat_conv4(c4))
        p3 = self._fuse(p4, self.lat_conv3(c3))
        p2 = self._fuse(p3, self.lat_conv2(c2))

        p4 = self.refiner4(p4)
        p3 = self.refiner3(p3)
        p2 = self.refiner2(p2)

        _, _, h, w = p2.shape

        # progressive resolution-aware supervision
        p5_g = p5 * self._scale_down(gabor_mask, 8)
        s5_ctx = F.relu(self.norm_context(self.context_dil2(p5_g)))
        s5_up = self._resize_to(s5_ctx, h, w)
        s5 = self._resize_to(F.relu(self.norm_semantic(self.semantic_refine(s5_up))), h, w)

        p4_g = p4 * self._scale_down(gabor_mask, 4)
        s4_ctx = F.relu(self.norm_context(self.context_dil4(p4_g)))
        s4_up = self._resize_to(s4_ctx, h, w)
        s4 = self._resize_to(F.relu(self.norm_semantic(self.semantic_refine(s4_up))), h, w)

        p3_g = p3 * self._scale_down(gabor_mask, 2)
        s3_ctx = F.relu(self.norm_context(self.context_dil6(p3_g)))
        s3 = self._resize_to(F.relu(self.norm_semantic(self.semantic_refine(s3_ctx))), h, w)

        p2_g = p2 * gabor_mask
        s2_ctx = F.relu(self.norm_context(self.context_dil8(p2_g)))
        s2 = F.relu(self.norm_semantic(self.semantic_refine(s2_ctx)))

        # Final fusion and prediction
        fused = s2 + s3 + s4 + s5
        fused = F.dropout2d(fused, p=0.2, training=self.training)

        B, C, H, W = fused.shape
        fused = fused.permute(0, 2, 3, 1)
        fused = self.classifier(fused)
        fused = fused.permute(0, 3, 1, 2)

        return {
            'final': fused,
            'aux_s3': self.supervise_s3(p3),
            'aux_s4': self.supervise_s4(p4),
            'aux_s5': self.supervise_s5(p5)
        }
