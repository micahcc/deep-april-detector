from typing import Optional, Callable
import math

import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation


def _sum(x: list[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class KeypointCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self,
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        """
        Args:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.bbox_xform_clip = bbox_xform_clip

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx1 = rel_codes[:, 0::8]
        dy1 = rel_codes[:, 1::8]
        dx2 = rel_codes[:, 2::8]
        dy2 = rel_codes[:, 3::8]
        dx3 = rel_codes[:, 4::8]
        dy3 = rel_codes[:, 5::8]
        dx4 = rel_codes[:, 6::8]
        dy4 = rel_codes[:, 7::8]

        pred_x1 = dx1 * widths[:, None] + ctr_x[:, None]
        pred_y1 = dy1 * heights[:, None] + ctr_y[:, None]
        pred_x2 = dx2 * widths[:, None] + ctr_x[:, None]
        pred_y2 = dy2 * heights[:, None] + ctr_y[:, None]
        pred_x3 = dx3 * widths[:, None] + ctr_x[:, None]
        pred_y3 = dy3 * heights[:, None] + ctr_y[:, None]
        pred_x4 = dx4 * widths[:, None] + ctr_x[:, None]
        pred_y4 = dy4 * heights[:, None] + ctr_y[:, None]

        pred_keypoints = torch.stack(
            (
                pred_x1,
                pred_y1,
                pred_x2,
                pred_y2,
                pred_x3,
                pred_y3,
                pred_x4,
                pred_y4,
            ),
            dim=2,
        ).flatten(1)
        return pred_keypoints


def _keypoint_loss(
    keypoint_coder: KeypointCoder,
    anchors_per_image: Tensor,
    matched_gt_boxes_per_image: Tensor,
    keypoint_regression_per_image: Tensor,
) -> Tensor:

    target_regression = keypoint_coder.encode_single(
        matched_gt_boxes_per_image, anchors_per_image
    )
    return torch.nn.functional.l1_loss(
        keypoint_regression_per_image, target_regression, reduction="sum"
    )


class RetinaNetKeypointHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(
        self,
        in_channels,
        num_anchors,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer)
            )
        self.conv = nn.Sequential(*conv)

        self.keypoint_coder = KeypointCoder()
        self.keypoint_reg = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.keypoint_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.keypoint_reg.bias)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (list[dict[str, Tensor]], dict[str, Tensor], list[Tensor], list[Tensor]) -> Tensor
        losses = []

        keypoint_regression = head_outputs["keypoint_regression"]

        for (
            targets_per_image,
            keypoint_regression_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, keypoint_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground keypoints
            matched_gt_keypoints_per_image = targets_per_image["keypoints"][
                matched_idxs_per_image[foreground_idxs_per_image]
            ]
            keypoint_regression_per_image = keypoint_regression_per_image[
                foreground_idxs_per_image, :
            ]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the loss
            losses.append(
                _keypoint_loss(
                    self.keypoint_coder,
                    anchors_per_image,
                    matched_gt_keypoints_per_image,
                    keypoint_regression_per_image,
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (list[Tensor]) -> Tensor
        all_keypoint_regression = []

        for features in x:
            keypoint_regression = self.conv(features)
            keypoint_regression = self.keypoint_reg(keypoint_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4, 2).
            N, _, H, W = keypoint_regression.shape
            keypoint_regression = keypoint_regression.view(N, -1, 8, H, W)
            keypoint_regression = keypoint_regression.permute(0, 3, 4, 1, 2)
            keypoint_regression = keypoint_regression.reshape(
                N, -1, 4, 2
            )  # Size=(N, HWA, 4)

            all_keypoint_regression.append(keypoint_regression)

        return torch.cat(all_keypoint_regression, dim=1)
