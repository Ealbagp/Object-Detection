# %%
import os
import cv2
import selectivesearch
import sys
# print current directory
print(os.getcwd())
sys.path.append("..")  # Go up one level, adjust as necessary

from module.utils import  (
    parse_xml, 
    prepare_proposals, 
    get_proposals, 
    calculate_iou, 
    load_image, 
    get_id,
    calc_recall, 
    calc_abo,
    from_xywh_to_min_max,
    visualize_image,
    resize_boxes)
from module.dataloader import (
    PotholeDataset
)

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import torch.utils.data as data
# Moving active directory to root folder to get module to work

img_path = "Potholes/annotated-images/"
anno_path = "Potholes/annotated-images/"
proposal_path = "tmp/"
# This does not scale. We should save annotation proposals in a file. 

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800
IMAGE_SIZE = (IMAGE_WIDTH,IMAGE_HEIGHT)


img_files = os.listdir(img_path)
proposal_files = os.listdir(proposal_path)

image_paths = np.array(list(filter(lambda file: file.endswith(".jpg"), img_files)))
label_paths = np.array(list(filter(lambda file: file.endswith(".xml"), img_files)))
proposal_paths = np.array(list(filter(lambda file: file.endswith(".xml"), proposal_files)))
# sort the files
image_paths = sorted(image_paths, key=get_id)
label_paths = sorted(label_paths, key=get_id)
proposal_paths = sorted(proposal_files, key=get_id)

# Limit to the first 100 images
image_paths = image_paths[:]
label_paths = label_paths[:]
proposal_paths = proposal_paths[:]

boxes = [parse_xml(anno_path + label_path) for label_path in label_paths]
images = [load_image(img_path + img) for img in image_paths]
proposals = [parse_xml(proposal_path + proposal) for proposal in proposal_paths]
gt_boxes = [resize_boxes(boxs, (image.shape[1], image.shape[0]), IMAGE_SIZE) for boxs, image in zip(boxes, images)]

# Proposals and gt boxes are in a (xmin,ymin,xmax, ymax) format
# We want to calculate the iou for each image using gt and proposals

def calculate_average_split(proposals, ground_truth_boxes, iou_threshold=0.5):
    total_proposals = 0
    overlapping_proposals = 0

    for proposal_set, gt_boxes in zip(proposals, ground_truth_boxes):
        for proposal in proposal_set:
            proposal_box = from_xywh_to_min_max(proposal)
            for gt_box in gt_boxes:
                iou = calculate_iou(proposal_box, gt_box)
                if iou >= iou_threshold:
                    overlapping_proposals += 1
                    break
            total_proposals += 1

    average_split = overlapping_proposals / total_proposals if total_proposals > 0 else 0
    return average_split

def calculate_split_for_each_image(proposals, ground_truth_boxes, iou_threshold=0.5):
    splits = []

    for proposal_set, gt_boxes in zip(proposals, ground_truth_boxes):
        total_proposals = 0
        overlapping_proposals = 0
        for proposal in proposal_set:
            proposal_box = from_xywh_to_min_max(proposal)
            for gt_box in gt_boxes:
                iou = calculate_iou(proposal_box, gt_box)
                if iou >= iou_threshold:
                    overlapping_proposals += 1
                    break
            total_proposals += 1
        split = overlapping_proposals / total_proposals if total_proposals > 0 else 0
        splits.append(split)

    return splits

def calculate_recal(proposals, ground_truth_boxes, iou_threshold=0.5):

    recall = 0
    for proposal_set, gt_boxes in zip(proposals, ground_truth_boxes):
        recall += calc_recall(proposal_set, gt_boxes, iou_threshold)

    recall = recall / len(proposal_files)
    return recall

# Calculate the average split for the first 100 images
average_split = calculate_average_split(proposals, gt_boxes)
print(f"Average split of proposals (IoU >= 0.5): {average_split}")

average_recall = calculate_recal(proposals, gt_boxes)
print(f"average recall for (IoU >= 0.5): {average_recall}")

potholdeDataset = PotholeDataset(img_path, proposal_path, anno_path, image_size=IMAGE_SIZE, iou_threshold=0.5)

dataloader = data.DataLoader(potholdeDataset, batch_size=1, shuffle=False)

recall = 0
for i, data in enumerate(dataloader):
    image, proposals, gt = data['image'], data['proposals'], data['gt_boxes']
    recall += calc_recall(proposals, gt, 0.5)
recall = recall / len(potholdeDataset)
# Calculate the split for each of the first 100 images
# splits = calculate_split_for_each_image(proposals, boxes)
# for idx, split in enumerate(splits):
#     print(f"Image {idx + 1}: Split of proposals overlapping with gt boxes (IoU >= 0.5): {split}")

# %%
