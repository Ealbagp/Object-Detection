import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
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
                 image_size=None, 
                 proposals_per_batch=20, 
                 balance=None,
                 transform=None,
                 iou_threshold=0.5,
                 proposal_size=None):
        self.images_path = images_path
        self.proposals_path = proposals_path
        self.gt_path = gt_path
        self.image_size = image_size
        self.proposals_per_batch = proposals_per_batch
        self.balance = balance
        self.transform = transform
        self.iou_threshold = iou_threshold
        self.proposal_size = proposal_size

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
        if self.image_size:
            for f in self.image_files:
                image = load_image(os.path.join(self.images_path, f))
                height, width = image.shape[:2]
                self.image_sizes.append((width, height))

        # Load and rescale ground truth boxes to match resolution of images
        self.gt = [parse_xml(os.path.join(self.gt_path, f)) for f in self.gt_files]
        if self.image_size:
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
            
        # Remove images which have no positive proposals using tensor indexing
        to_keep = ~torch.tensor([labels.sum() == 0 for labels in self.labels])
        self.image_files = list(np.array(self.image_files)[to_keep.numpy()])
        self.gt = list(np.array(self.gt, dtype=object)[to_keep.numpy()])
        self.proposals = list(np.array(self.proposals, dtype=object)[to_keep.numpy()])
        self.labels = list(np.array(self.labels, dtype=object)[to_keep.numpy()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and resize image
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = load_image(image_path)
        if self.image_size:
           image = cv2.resize(image, self.image_size)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        gt_boxes = self.gt[idx]
        proposals = self.proposals[idx]
        labels = self.labels[idx]

        if self.balance:
            # Separate positive and negative proposals
            pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

            # Determine number of samples based on balance
            num_pos_samples = int(self.balance * self.proposals_per_batch)
            num_neg_samples = self.proposals_per_batch - num_pos_samples

            # Adjust for availability
            num_pos_samples = min(num_pos_samples, len(pos_indices))
            num_neg_samples = min(num_neg_samples, len(neg_indices))

            # Sample indices
            pos_samples = pos_indices[torch.randperm(len(pos_indices))[:num_pos_samples]]
            neg_samples = neg_indices[torch.randperm(len(neg_indices))[:num_neg_samples]]
            selected_indices = torch.cat([pos_samples, neg_samples])
        else:
            num_samples = min(len(labels), self.proposals_per_batch)
            selected_indices = torch.randperm(len(labels))[:num_samples]

        # Select balanced proposals and labels
        proposals = proposals[selected_indices]
        labels = labels[selected_indices]

        # Cut out proposals from the image and resize them
        proposal_images = []
        for proposal in proposals:
            x, y, w, h = proposal.tolist()
            x, y, w, h = int(x), int(y), int(w), int(h)
            proposal_img = image[:, y:y+h, x:x+w]
            if self.proposal_size:
                proposal_img = F.interpolate(proposal_img.unsqueeze(0), size=self.proposal_size, mode='bilinear', align_corners=False).squeeze(0)
            proposal_images.append(proposal_img)
        
        proposal_images = torch.stack(proposal_images)  # Shape: (proposals_per_batch, C, H, W)
        
        # Apply transform if specified
        if self.transform:
            proposal_images = self.transform(proposal_images)

        dict_data = {
            'image': image,
            'proposals': proposals,
            'labels': labels,
            'gt_boxes': gt_boxes,
            'proposal_images': proposal_images
        }
        
        return dict_data

