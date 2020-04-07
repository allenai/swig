import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from global_utils.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from global_utils.anchors import Anchors
from global_utils import losses
# from lib.nms.pth_nms import pth_nms
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import ops
import random
from torch.autograd import Variable
#from bbox_features import BoxFeatures
from global_utils.resnet import ResNet



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, C3, C4, C5):

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
        # return [P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=5, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)


    def forward(self, x):
        batch_size, channels, width, height = x.shape

        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)

        out = self.act3(out)
        out = self.conv4(out)
        out1 = self.act4(out)
        out = self.output(out1)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(batch_size, -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=5, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.spatial_conv = nn.Conv2d(2, 64, kernel_size=1)
        self.bbox_conv = nn.Conv2d(4, 64, kernel_size=1)
        self.mask_conv = nn.Conv2d(1, 64, kernel_size=1)

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.features_linear = nn.Linear(feature_size, 1)
        self.output_retina = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act_retina = nn.Sigmoid()


    def forward(self, x):

        batch_size, channels, width, height = x.shape

        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out_0 = self.conv4(out)
        # BBox Binary Logit
        bbox_exists = self.pool(out_0).squeeze()
        bbox_exists = self.features_linear(bbox_exists)

        # Classification Branch
        out = self.act4(out_0)
        out1 = self.output_retina(out)
        out1 = self.output_act_retina(out1)
        out1 = out1.permute(0, 2, 3, 1)  # out is B x C x W x H, with C = n_classes + n_anchors
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(batch_size, -1, self.num_classes), bbox_exists


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    def forward(self, x, target):

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ResNet_RetinaNet_RNN(nn.Module):

    def __init__(self, num_classes, num_nouns, block, layers, parser, cat_features=False):
        self.inplanes = 64
        super(ResNet_RetinaNet_RNN, self).__init__()

        #self.bbox_features = BoxFeatures()

        self.num_classes = num_classes
        self.num_nouns = num_nouns

        self._init_resnet(block, layers)
        self.fpn = PyramidFeatures(self.fpn_sizes[0], self.fpn_sizes[1], self.fpn_sizes[2])

        self.hidden_size = parser.hidden_size
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()
        self.cat_features = cat_features

        self._convs_and_bn_weights()

        #self.feature_extractor = ResNet()

        # verb predictor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024*2, 504)

        self.vocab_linear = nn.Linear(2304, 1024)
        self.vocab_linear_2 = nn.Linear(1024, num_nouns)
        self.loss_function = LabelSmoothing(0.2)

        self.relu = nn.ReLU()

        # init embeddings
        self.verb_embeding = nn.Embedding(504, 256)
        self.noun_embedding = nn.Embedding(num_nouns, 512)

        self.regressionModel = RegressionModel(768)
        self.classificationModel = ClassificationModel(768, num_classes=num_classes, feature_size=256)

        # init rnn and rnn weights
        self.rnn = nn.LSTMCell(4864, self.hidden_size)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

        self.rnn_linear = nn.Linear(self.hidden_size, 256)

        # fill class/reg branches with weights
        prior = 0.01

        self.classificationModel.output_retina.weight.data.fill_(0)
        self.classificationModel.output_retina.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()
        self.verb_loss_function = nn.CrossEntropyLoss()


    def forward(self, img_batch, annotations, verb, widths, heights, epoch_num, detach_resnet=False, use_gt_nouns=False, use_gt_verb=False, return_local_features=False):

        batch_size = img_batch.shape[0]

        # Extract Visual Features
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)

        if detach_resnet:
            with torch.no_grad():
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
        else:
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

        image_predict = self.avgpool(x4).squeeze()
        verb_word = self.verb_embeding(verb.long())

        # Get feature pyramid
        features = self.fpn(x2, x3, x4)
        anchors = self.anchors(img_batch)
        features.pop(0)  # SARAH - remove feature batch

        # init LSTM inputs
        hx, cx = torch.zeros(batch_size, self.hidden_size).cuda(), torch.zeros(batch_size, self.hidden_size).cuda()
        previous_word = torch.zeros(batch_size, 512).cuda(x.device)
        roi_features = torch.zeros(batch_size, 2048).cuda(x.device)

        # init losses
        all_class_loss = 0
        all_bbox_loss = 0
        all_reg_loss = 0
        noun_loss = 0

        if self.training:
            class_list = []
            reg_list = []
            bbox_pred_list = []
        else:
            noun_predicts = []
            bbox_predicts = []
            bbox_exist_list = []


        if return_local_features:
            local_features = []

        for i in range(6):
            rnn_input = torch.cat((image_predict, verb_word, previous_word, roi_features), dim=1)
            hx, cx = self.rnn(rnn_input, (hx, cx))
            rnn_output = self.rnn_linear(hx)
            just_rnn = [rnn_output.view(batch_size, 256, 1, 1).expand((batch_size, 256, f.shape[2], f.shape[3])) for f in features]

            bbox_exist, regression, classification, top_class_per_box = self.class_and_reg_branch(batch_size, rnn_output, features, just_rnn)

            if self.training and use_gt_nouns:
                previous_boxes = annotations[:, i, :4]
            else:
                transformed_anchors = self.regressBoxes(anchors.detach(), regression.detach())
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
                bounding_boxes_indices = torch.argmax(top_class_per_box, dim=1)
                previous_boxes = transformed_anchors[torch.arange(batch_size), bounding_boxes_indices.squeeze()]
                previous_boxes[bbox_exist < 0] = -1

            roi_features = self.get_local_features(x4, previous_boxes, img_batch.shape[2], img_batch.shape[3])
            noun_pred = torch.cat((roi_features, rnn_output), dim=1)
            noun_pred = self.vocab_linear(noun_pred)
            noun_pred = self.vocab_linear_2(self.relu(noun_pred))
            classification_guess = torch.argmax(noun_pred, dim=1)

            if return_local_features:
                local_features.append(roi_features)

            if self.training:
                for noun_index in range(4, 7):
                    noun_gt = annotations[torch.arange(batch_size), i, noun_index]
                    noun_loss += self.loss_function(noun_pred, noun_gt.squeeze().long().cuda())

            if self.training and use_gt_nouns:
                ground_truth_1 = self.noun_embedding(annotations[:, i, 4].long())
                ground_truth_2 = self.noun_embedding(annotations[:, i, 5].long())
                ground_truth_3 = self.noun_embedding(annotations[:, i, 6].long())
                previous_word = torch.stack([ground_truth_1, ground_truth_2, ground_truth_3]).mean(dim=0)
            else:
                previous_word = self.noun_embedding(classification_guess)

            if self.training:
                class_list.append(classification)
                reg_list.append(regression)
                bbox_pred_list.append(bbox_exist)
            else:
                bbox_predicts.append(previous_boxes)
                noun_predicts.append(classification_guess)
                bbox_exist_list.append(bbox_exist)

        if return_local_features:
            all_local_features = torch.stack(local_features).transpose(0,1)

        if self.training:
            anns = annotations[:, :, :].unsqueeze(1)
            class_all = torch.cat([c.unsqueeze(1) for c in class_list], dim=1)
            reg_all = torch.cat([c.unsqueeze(1) for c in reg_list], dim=1)
            bbox_ex_all = torch.cat([c.unsqueeze(1) for c in bbox_pred_list], dim=1)

            class_loss, reg_loss, bbox_loss = self.focalLoss(class_all, reg_all, anchors, bbox_ex_all, anns.squeeze())
            all_class_loss += class_loss
            all_reg_loss += reg_loss
            all_bbox_loss += bbox_loss
            return all_class_loss, all_reg_loss, all_bbox_loss, noun_loss

        else:
            if return_local_features:
                return verb, noun_predicts, bbox_predicts, bbox_exist_list, all_local_features

            return verb, noun_predicts, bbox_predicts, bbox_exist_list


    def class_and_reg_branch(self, batch_size, rnn_output, features, just_rnn):
        rnn_feature_mult = [rnn_output.view(batch_size, 256, 1, 1).expand(feature.shape) * feature for feature in
                            features]
        rnn_feature_shapes = [torch.cat([just_rnn[ii], features[ii], rnn_feature_mult[ii]], dim=1) for ii in
                              range(len(features))]
        regression = torch.cat([self.regressionModel(rnn_feature_shapes[ii]) for ii in range(len(rnn_feature_shapes))],
                               dim=1)
        classifications = []
        bbox_exist = []
        for ii in range(len(rnn_feature_shapes)):
            classication = self.classificationModel(rnn_feature_shapes[ii])
            classifications.append(classication[0])
            bbox_exist.append(classication[1])

        if len(bbox_exist[0].shape) == 1:
            bbox_exist = [c.unsqueeze(0) for c in bbox_exist]
        bbox_exist = torch.cat([c for c in bbox_exist], dim=1)
        bbox_exist = torch.max(bbox_exist, dim=1)[0]
        # get max from K x A x W x H to get max classificiation and bbox
        classification = torch.cat([c for c in classifications], dim=1)
        scores = torch.max(classification, dim=2, keepdim=True)[0]

        return bbox_exist, regression, classification, scores



    def get_local_features(self, features, boxes, picture_width, picture_height):
        features_heights = features.shape[3]
        features_width = features.shape[2]
        boxes_copy = boxes.clone()
        boxes_copy[:, 0] = (boxes_copy[:, 0]*features_heights) / picture_height
        boxes_copy[:, 2] = (boxes_copy[:, 2]*features_heights) / picture_height
        boxes_copy[:, 1] = (boxes_copy[:, 1]*features_width) / picture_width
        boxes_copy[:, 3] = (boxes_copy[:, 3]*features_width) / picture_width
        batch = torch.arange(boxes_copy.shape[0]).unsqueeze(1).cuda().float()
        box_input = torch.cat((batch, boxes_copy), dim=1)
        roi_align_output = ops.roi_align(features, box_input, (1,1)).squeeze()
        roi_align_output[boxes[:, 0] == -1, :] = F.adaptive_avg_pool2d(features, (1,1)).squeeze()[boxes[:, 0] == -1, :]
        roi_align_output= roi_align_output.squeeze()

        if len(roi_align_output.shape) == 1:
            roi_align_output = roi_align_output.unsqueeze(0)

        return roi_align_output

    def _init_resnet(self, block, layers):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            self.fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels,
                              self.layer3[layers[2] - 1].conv2.out_channels,
                              self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            self.fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                              self.layer3[layers[2] - 1].conv3.out_channels,
                              self.layer4[layers[3] - 1].conv3.out_channels]


    def _convs_and_bn_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def resnet50(num_classes, num_nouns, parser, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_RetinaNet_RNN(num_classes, num_nouns, Bottleneck, [3, 4, 6, 3], parser,  **kwargs)


    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'], model_dir='.')
        print("state dict")
        x = nn.Linear(1024*2, 504)
        state_dict['fc.weight'] = x.weight
        state_dict['fc.bias'] = x.bias
        model.load_state_dict(state_dict, strict=False)
    return model

