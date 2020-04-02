from collections import defaultdict
import pdb
from PIL import Image
import numpy as np
import cv2

class BboxEval:
    def __init__(self):
        self.per_verb_occ_bboxes = defaultdict(float)
        self.per_verb_all_correct_bboxes = defaultdict(float)
        self.per_verb_roles_bboxes = defaultdict(float)
        self.per_verb_roles_correct_bboxes = defaultdict(float)

        self.per_verb_occ = defaultdict(float)
        self.per_verb_all_correct = defaultdict(float)
        self.per_verb_roles = defaultdict(float)
        self.per_verb_roles_correct = defaultdict(float)

        self.all_verbs = 0.0
        self.correct_verbs = 0.0


    def verb(self):
        return self.correct_verbs/self.all_verbs


    def value_all(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for verb in self.per_verb_occ:
            sum_value_all += float(self.per_verb_all_correct[verb])/float(self.per_verb_occ[verb])
            total_value_all += 1.0
        return sum_value_all/total_value_all


    def value(self):
        sum_value = 0.0
        total_value = 0.0
        for verb in self.per_verb_roles:
            sum_value += float(self.per_verb_roles_correct[verb]) / float(self.per_verb_roles[verb])
            total_value += 1.0
        return sum_value / total_value


    def value_all_bbox(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for verb in self.per_verb_occ_bboxes:
            sum_value_all += float(self.per_verb_all_correct_bboxes[verb])/float(self.per_verb_occ_bboxes[verb])
            total_value_all += 1.0
        return sum_value_all/total_value_all


    def value_bbox(self):
        sum_value = 0.0
        total_value = 0.0
        for verb in self.per_verb_roles_bboxes:
            sum_value += float(self.per_verb_roles_correct_bboxes[verb]) / float(self.per_verb_roles_bboxes[verb])
            total_value += 1.0
        return sum_value / total_value


    def update(self, pred_verb, pred_nouns, pred_bboxes, gt_verb, gt_nouns, gt_bboxes, verb_order, verb_correct):
        order = verb_order[gt_verb]["order"]

        self.all_verbs += 1.0
        self.per_verb_occ[gt_verb] += 1.0
        self.per_verb_occ_bboxes[gt_verb] += 1.0

        self.per_verb_roles[gt_verb] += len(order)
        self.per_verb_roles_bboxes[gt_verb] += len(order)

        if len(pred_nouns) == 0:
            pdb.set_trace()

        if pred_verb == gt_verb:
            self.correct_verbs += verb_correct
        value_all_bbox = 1.0
        value_all = 1.0

        value = []

        if True:
            for i in range(len(order)):
                if pred_nouns[i] in gt_nouns[i]:
                    self.per_verb_roles_correct[gt_verb] += 1.0
                    value.append(1)
                else:
                    value_all = 0.0
                    value.append(0)
                if pred_nouns[i] in gt_nouns[i] and (self.bb_intersection_over_union(pred_bboxes[i], gt_bboxes[i])):
                    self.per_verb_roles_correct_bboxes[gt_verb] += 1.0
                else:
                    value_all_bbox = 0.0
            self.per_verb_all_correct_bboxes[gt_verb] += value_all_bbox
            self.per_verb_all_correct[gt_verb] += value_all

        return value



    def bb_intersection_over_union(self, boxA, boxB):
        if boxA is None and boxB is None:
            return True
        if boxA is None or boxB is None:
            return False
        #boxB = [b / 2.0 for b in boxB]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou > 0.5:
            return True
        else:
            return False