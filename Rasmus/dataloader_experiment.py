#%%
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms.v2 as transforms
import os
import sys
import cv2
import matplotlib.pyplot as plt
sys.path.append("..")  # Go up one level, adjust as necessary

from module.utils import (
    parse_xml, 
    calculate_iou, 
    get_id,
    load_image,
    resize_boxes    
    )

def visualize_image(image, boxes,labels, proposals=None):
    # Adjust ground truth boxes according to the scale
    
    # Convert color for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw ground truth boxes in blue
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    # Draw Selective Search proposals in green if provided
    if proposals is not None:
        for (x, y, w, h), label in zip(proposals,labels):
            # Adjust Selective Search boxes according to the scale


            x, y, w, h = int(x), int(y), int(w), int(h)
            if label == 1:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # else:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                
            # cv2.putText(image, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def label_proposals(proposals, gt, iou_threshold=0.5):
    labels = np.zeros(len(proposals))
    for i, proposal in enumerate(proposals):
        # transform proposal to (xmin, ymin, xmax, ymax)
        proposal = (proposal[0], proposal[1], proposal[0] + proposal[2], proposal[1] + proposal[3])
        max_iou = 0
        for gt_box in gt:
            iou = calculate_iou(proposal, gt_box)
            if iou > max_iou:
                max_iou = iou
        labels[i] = 1 if max_iou >= iou_threshold else 0
    return labels


def load_data(images_path, proposals_path, gt_path, image_size=(800, 800)):
    data = []
    
    images = os.listdir(images_path)
    gt = os.listdir(gt_path)
    proposals = os.listdir(proposals_path)

    image_paths = np.array(list(filter(lambda file: file.endswith(".jpg"), images)))
    gt_paths = np.array(list(filter(lambda file: file.endswith(".xml"), gt)))
    proposals_paths = np.array(list(filter(lambda file: file.endswith(".xml"), proposals)))

    image_paths = sorted(image_paths, key=get_id)
    gt_paths = sorted(gt_paths, key=get_id)
    proposals_paths = sorted(proposals_paths, key=get_id)
    
    for image, gt, proposal in zip(image_paths, gt_paths, proposals_paths):
        img = load_image(os.path.join(images_path, image))
        gt = parse_xml(os.path.join(gt_path, gt))
        gt = resize_boxes(gt, (img.shape[1], img.shape[0]), image_size)
        img = cv2.resize(img, image_size)
        proposals = parse_xml(os.path.join(proposals_path, proposal))

        data.append({'image': img, 'proposals': proposals, 'gt': gt})
    return data

# Pothole Dataset Class
class PotholeDataset(data.Dataset):
    def __init__(self, images_path, proposals_path, gt_path, transform=None):
        self.images_path = images_path
        self.ground_truth_path = gt_path
        self.annotations_path = proposals_path
        self.transform = transform

    def get_proposals(self,idx, count):
        proposal_data = []
        
        gt = parse_xml(self.ground_truth_path)
        

        return proposal_data

    def get_proposals(self, image):
        # Placeholder function for proposal generation (to be implemented)
        proposals = []  # Replace with actual proposal generation code
        return proposals[:self.proposals_per_image]

    def __len__(self):
        return len(self.proposal_data)

    def __getitem__(self, idx):
        image_filename = data['image_filename']
        proposal = data['proposal']
        label = data['label']

        # Load image
        image_path = os.path.join(self.images_path, image_filename)
        image = cv2.imread(image_path)
        
        if self.transform:
            image = self.transform(image)

        return image, proposal, label
    
# test of dataloader
# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script directory
gt_path = os.path.normpath(os.path.join(script_dir, "../Potholes/annotated-images/"))
images_path = os.path.normpath(os.path.join(script_dir, "../Potholes/annotated-images/"))
proposals_path = os.path.normpath(os.path.join(script_dir, "tmp/"))

print(images_path)
print(gt_path)
print(proposals_path)
transform = transforms.Compose([transforms.ToTensor()])

#%%
# current directory
print(os.getcwd())

assert os.path.exists(images_path), "Images path does not exist"
assert os.path.exists(gt_path), "Ground truth path does not exist"
assert os.path.exists(proposals_path), "Proposals path does not exist"

data = load_data(images_path, proposals_path, gt_path)
# %%

# load and display an image
image = data[0]['image']
proposals = data[0]['proposals']
gt = data[0]['gt']
labels = label_proposals(proposals, gt)

# %%
print(gt)
print(proposals[:10])

# %%

visualize_image(image, gt, labels=labels,proposals=proposals)
# %%


