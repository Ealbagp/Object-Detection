import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from module.utils import (
    parse_xml, 
    calculate_iou, 
    get_id,
    load_image,
    resize_boxes,
    from_xywh_to_min_max
)

class PotholeDataset(Dataset):
    def __init__(self, 
                 images_path, 
                 proposals_path, 
                 gt_path, 
                 image_size=(800, 800), 
                 proposals_per_batch=20, 
                 balance=None,
                 transform=None,
                 iou_threshold=0.5):
        self.images_path = images_path
        self.proposals_path = proposals_path
        self.gt_path = gt_path
        self.image_size = image_size
        self.proposals_per_batch = proposals_per_batch
        self.balance = balance
        self.transform = transform
        self.iou_threshold = iou_threshold

        # Get sorted list of files
        self.image_files = sorted(
            [f for f in os.listdir(self.images_path) if f.endswith('.jpg')],
            key=get_id
        )
        self.gt_files = sorted(
            [f for f in os.listdir(self.gt_path) if f.endswith('.xml')],
            key=get_id
        )
        self.proposal_files = sorted(
            [f for f in os.listdir(self.proposals_path) if f.endswith('.xml')],
            key=get_id
        )
        
        # Get image sizes to resize ground truth boxes
        self.image_sizes = []
        for f in self.image_files:
            image = load_image(os.path.join(self.images_path, f))
            height, width = image.shape[:2]
            self.image_sizes.append((width, height))

        # Load and rescale ground truth boxes to match resolution of images
        self.gt = [parse_xml(os.path.join(self.gt_path, f)) for f in self.gt_files]
        self.gt = [resize_boxes(gt, original_size, self.image_size) for gt, original_size in zip(self.gt, self.image_sizes)]
        self.gt = [torch.tensor(gt, dtype=torch.int32) for gt in self.gt]
        
        # Load proposals
        self.proposals = [parse_xml(os.path.join(self.proposals_path, f)) for f in self.proposal_files]
        self.proposals = [torch.tensor(proposal, dtype=torch.int32) for proposal in self.proposals]
        
        # Ensure the number of images, ground truths, and proposals are the same
        assert len(self.image_files) == len(self.gt) == len(self.proposals), "Dataset files mismatch"
        
        # Label proposals based on IoU with ground truth boxes
        self.labels = []
        for gt_boxes, proposals in zip(self.gt, self.proposals):
            labels = []
            for proposal in proposals:
                max_iou = 0
                for gt_box in gt_boxes:
                    prop = from_xywh_to_min_max(proposal)
                    iou = calculate_iou(prop, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                label = 1 if max_iou >= self.iou_threshold else 0
                labels.append(label)
            labels = torch.tensor(labels, dtype=torch.long)
            self.labels.append(labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and resize image
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = load_image(image_path)
        image = cv2.resize(image, self.image_size)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        gt_boxes = self.gt[idx]
        proposals = self.proposals[idx]
        labels = self.labels[idx]

        if self.balance:
            # Separate positive and negative proposals
            pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

            # Determine number of samples per class
            num_samples_per_class = min(len(pos_indices), len(neg_indices), self.proposals_per_batch // 2, 2)

            # Sample indices
            pos_samples = pos_indices[torch.randperm(len(pos_indices))[:num_samples_per_class]]
            neg_samples = neg_indices[torch.randperm(len(neg_indices))[:num_samples_per_class]]
            selected_indices = torch.cat([pos_samples, neg_samples])
        else:
            num_samples = min(len(labels), self.proposals_per_batch)
            selected_indices = torch.randperm(len(labels))[:num_samples]

        # Select balanced proposals and labels
        proposals = proposals[selected_indices]
        labels = labels[selected_indices]

        if self.transform:
            image = self.transform(image)

        dict_data = {
            'image': image,
            'proposals': proposals,
            'labels': labels,
            'gt_boxes': gt_boxes
        }
        
        return dict_data

