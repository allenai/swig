import numpy as np
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import time


@torch.jit.script
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    return IoU

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.loss_function = nn.BCELoss().cuda()


    def forward(self, classifications, regressions, anchors, bbox_exist_prediction, annotations):
        all_time = 0
        IoU_time = 0
        sum_time = 0
        start = time.time()

        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]

        classification_losses = []
        regression_losses = []
        bbox_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights


        classifications = torch.clamp(classifications, 1e-4, 1.0 - 1e-4)

        bbox_exists_tensor = annotations[:, :, 0] != -1
        bbox_loss = F.binary_cross_entropy_with_logits(bbox_exist_prediction, bbox_exists_tensor.float())


        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            iou_start = time.time()

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            IoU = IoU.transpose(0, 1)

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU, 0.4), :] = 0
            positive_indices = torch.ge(IoU, 0.5)
            num_positive_anchors = positive_indices.sum(dim=1)

            #assigned_annotations = bbox_annotation[IoU_argmax, :]
            targets[positive_indices, :] = 0

            #assign for the 3 different classes
            for bbox_number in range(6):
                #targets[bbox_number, positive_indices[bbox_number], bbox_annotation[bbox_number, 7].long()] = 1
                targets[bbox_number, positive_indices[bbox_number], :] = 1


            alpha_factor = torch.ones(targets.shape).cuda() * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            IoU_time += time.time() - iou_start

            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())


            classification_losses.append(cls_loss.sum(dim=1).sum(dim=1)/torch.clamp(num_positive_anchors.float(), min=1.0))
            #positive_indices = positive_indices.long()
            #positive_indices = positive_indices.bool()



            # compute the loss for regression
            for i in range(6):

                if num_positive_anchors[i] > 0 and bbox_exists_tensor[j, i]:

                    # x = torch.arange(len(positive_indices[i]))
                    # x = x[positive_indices[i]]

                    #assigned_annotations = assigned_annotations[positive_indices[:, i], :]
                    assigned_annotations = bbox_annotation[i, :].long().unsqueeze(0).float()

                    anchor_widths_pi = anchor_widths[positive_indices[i]]

                    anchor_heights_pi = anchor_heights[positive_indices[i]]
                    anchor_ctr_x_pi = anchor_ctr_x[positive_indices[i]]
                    anchor_ctr_y_pi = anchor_ctr_y[positive_indices[i]]

                    gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                    gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                    gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                    # clip widths to 1
                    gt_widths  = torch.clamp(gt_widths, min=1)
                    gt_heights = torch.clamp(gt_heights, min=1)

                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)
                    targets_dh = torch.log(gt_heights / anchor_heights_pi)

                    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                    targets = targets.t()

                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                    #negative_indices = 1 - positive_indices[i]

                    regression_diff = torch.abs(targets - regression[i, positive_indices[i], :])

                    regression_loss = torch.where(
                        torch.le(regression_diff, 1.0 / 9.0),
                        0.5 * 9.0 * torch.pow(regression_diff, 2),
                        regression_diff - 0.5 / 9.0
                    )
                    regression_losses.append(regression_loss.mean())
                else:
                    regression_losses.append(torch.tensor(0).float().cuda())

        class_loss = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        reg_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)
        #bbox_loss = torch.stack(bbox_losses).mean(dim=0, keepdim=True)

        return class_loss, reg_loss, bbox_loss