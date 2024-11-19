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
    from_xywh_to_min_max,
    parse_xml_with_labels
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
                 proposal_size=(64, 64), 
                 split_ratios=(0.7, 0.15, 0.15), 
                 split="train"):
        self.images_path = images_path
        self.images_path = images_path
        self.proposals_path = proposals_path
        self.gt_path = gt_path
        self.image_size = image_size
        self.proposals_per_batch = proposals_per_batch
        self.balance = balance
        self.transform = transform
        self.iou_threshold = iou_threshold
        self.proposal_size = proposal_size
        self.split_ratios = split_ratios  # Ratios for train/val/test splits
        self.split = split

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
        
        # Load proposals and labels directly
        self.proposals_labels = [parse_xml_with_labels(os.path.join(self.proposals_path, f)) 
                                 for f in self.proposal_files]
        self.proposals = [torch.tensor(pl[0], dtype=torch.int32) for pl in self.proposals_labels]
        self.labels = [torch.tensor(pl[1], dtype=torch.long) for pl in self.proposals_labels]

        # Ensure the number of images, ground truths, proposals, and labels are the same
        assert len(self.image_files) == len(self.gt) == len(self.proposals) == len(self.labels), "Dataset files mismatch"
        
        # Remove images which have no positive proposals using tensor indexing
        to_keep = ~torch.tensor([labels.sum() == 0 for labels in self.labels])
        self.image_files = list(np.array(self.image_files)[to_keep.numpy()])
        self.gt = list(np.array(self.gt, dtype=object)[to_keep.numpy()])
        self.proposals = list(np.array(self.proposals, dtype=object)[to_keep.numpy()])
        self.labels = list(np.array(self.labels, dtype=object)[to_keep.numpy()])

        # Apply train/val/test split
        self.__apply_split()

    def __apply_split(self):
        total_samples = len(self.image_files)
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples)

        if self.split == "train":
            indices = range(0, train_end)
        elif self.split == "val":
            indices = range(train_end, val_end)
        elif self.split == "test":
            indices = range(val_end, total_samples)
        else:
            raise ValueError(f"Invalid split name '{self.split}', expected 'train', 'val', or 'test'.")

        self.image_files = [self.image_files[i] for i in indices]
        self.gt = [self.gt[i] for i in indices]
        self.proposals = [self.proposals[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and resize image
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = load_image(image_path)
        if self.image_size:
           image = cv2.resize(image, self.image_size)
        
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        
        #image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

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
        labels = labels[selected_indices].long() # Class indices should be long for torch to work.

        # Cut out proposals from the image and resize them
        proposal_images = []
        for proposal in proposals:
            x, y, w, h = proposal.tolist()
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Ensure the crop is within image bounds
            img_height, img_width = image.shape[0], image.shape[1]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)

            # Extract proposal and prepare for transforms
            proposal_img = image[y:y+h, x:x+w, :]
            
            # Apply transforms (includes ToTensor which handles CHW conversion)
            if self.transform:
                proposal_img = self.transform(proposal_img)  # Now shape is [3, H, W]
            else:
                proposal_img = torch.tensor(proposal_img, dtype=torch.float32).permute(2, 0, 1)
            
            # Resize if needed
            if self.proposal_size:
                proposal_img = F.interpolate(
                    proposal_img.unsqueeze(0), 
                    size=self.proposal_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            proposal_images.append(proposal_img)

        # Stack the correctly shaped tensors
        proposal_images = torch.stack(proposal_images)  # Final shape: [N, 3, H, W]
        # print(f"Proposal images shape AFTER stack: {proposal_images.shape}")
        
        # Apply transform if specified
        

        dict_data = {
            'image': image,
            'proposals': proposals,
            'labels': labels,
            'gt_boxes': gt_boxes,
            'proposal_images': proposal_images
        }
        
        return {
            'image': image,
            'proposals': proposals,
            'labels': labels,
            'gt_boxes': gt_boxes,
            'proposal_images': proposal_images
        }

