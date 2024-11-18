import os
import cv2
import selectivesearch
import sys

sys.path.append("..")  # Go up one level, adjust as necessary

from module.utils import  from_xywh_to_min_max,parse_xml, prepare_proposals,get_proposals, calculate_iou, load_image, get_id,calc_recall, calc_abo, resize_box
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
#import torch 
from torch.utils.data import Dataset

def create_labeled_proposals_file(proposal_file, gt_file, image_path, output_path, iou_threshold=0.5):
    """
    Load proposals from one file and create a new file with labels based on IoU.
    
    Args:
        proposal_file: Path to XML file containing proposals
        gt_file: Path to XML file containing ground truth boxes
        image_path: Path to the corresponding image
        output_path: Where to save the labeled proposals XML
        iou_threshold: IoU threshold for positive samples
    """
    # Load proposals and ground truth boxes
    proposals = parse_xml(proposal_file)
    gt_boxes = parse_xml(gt_file)
    
    # Calculate labels
    labels = []
    for proposal in proposals:
        max_iou = 0
        prop = from_xywh_to_min_max(proposal)
        for gt_box in gt_boxes:
            iou = calculate_iou(prop, gt_box)
            max_iou = max(max_iou, iou)
        labels.append(1 if max_iou >= iou_threshold else 0)
    
    # Create XML structure
    root = ET.Element("annotation")
    
    # Add image info
    folder = ET.SubElement(root, "folder")
    folder.text = "Labeled_Proposals"
    
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)
    
    # Add image size info
    image = cv2.imread(image_path)
    height, width, depth = image.shape
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    
    # Add proposals with labels
    for proposal, label in zip(proposals, labels):
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = "pothole" if label == 1 else "background"
        
        # Add label
        label_elem = ET.SubElement(obj, "label")
        label_elem.text = str(label)
        
        # Add bounding box
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        
        xmin.text = str(int(proposal[0]))
        ymin.text = str(int(proposal[1]))
        xmax.text = str(int(proposal[2]))
        ymax.text = str(int(proposal[3]))
    
    # Save to file
    tree = ET.ElementTree(root)
    tree.write(output_path)

def process_all_proposals(proposals_dir, gt_dir, images_dir, output_dir, iou_threshold=0.5):
    """Process all proposal files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for proposal files in: {os.path.abspath(proposals_dir)}")
    proposal_files = sorted([f for f in os.listdir(proposals_dir) if f.endswith('_proposals.xml')], key=get_id)
    print(f"Found {len(proposal_files)} proposal files")
    
    for prop_file in proposal_files:
        # Get numeric ID without the suffix
        image_id = str(get_id(prop_file.replace('_proposals.xml', '')))
        
        # Create paths with correct filename format (img-{id})
        prop_path = os.path.abspath(os.path.join(proposals_dir, prop_file))
        gt_path = os.path.abspath(os.path.join(gt_dir, f"img-{image_id}.xml"))
        img_path = os.path.abspath(os.path.join(images_dir, f"img-{image_id}.jpg"))
        out_path = os.path.abspath(os.path.join(output_dir, f"img-{image_id}_labeled_proposals.xml"))
        
        # Verify all required files exist
        if not os.path.exists(prop_path):
            print(f"Proposal file not found: {prop_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Ground truth file not found: {gt_path}")
            continue
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            continue
            
        print(f"Processing image {image_id}")
        create_labeled_proposals_file(prop_path, gt_path, img_path, out_path, iou_threshold)

if __name__ == "__main__":
    # Use direct paths without ../
    img_path = "../Potholes/annotated-images/"
    anno_path = "../Potholes/annotated-images/"
    proposals_dir = "tmp/"
    output_dir = "tmp_labeled/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    process_all_proposals(proposals_dir, anno_path, img_path, output_dir)

