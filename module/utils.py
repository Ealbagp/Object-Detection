import os
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import random

# Helper function to parse the XML file for ground truth bounding boxes
def parse_xml(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


# Function to run Selective Search and obtain proposals
def get_proposals(image, num_proposals, scale = 25, sigma=0.8, min_size=300):
    _, regions = selectivesearch.selective_search(image, scale=scale, sigma=sigma, min_size=min_size)
    
    filtered_regions = list(filter(lambda r:  r['rect'][2] > 20 and r['rect'][3] > 20, regions))
    
    # Ensure the number of proposals does not exceed available regions
    num_proposals = min(num_proposals, len(filtered_regions))
    
    # Randomly select unique indexes from the filtered regions
    selected_regions = random.sample(filtered_regions, num_proposals)
    
    # Extract and return the (x, y, w, h) proposals
    proposals = [(r['rect'][0], r['rect'][1], r['rect'][2], r['rect'][3]) for r in selected_regions]
    
    return proposals

# Constants
IOU_THRESHOLD = 0.5  # For recall calculation
TARGET_PROPOSALS = range(10, 200, 10)  # Range of number of proposals to test (adjust as needed)

# Function to calculate the IoU between to proposals(two boxes)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxA_Area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_Area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxA_Area + boxB_Area - interArea)
    return iou

# Function to evaluate proposals on a single image using recall
def calc_recall(proposals, ground_truth_boxes, iou_threshold):
    recalled_boxes = 0
    for gt_box in ground_truth_boxes:
        max_iou = 0  # Variable to store the maximum IoU for this ground truth box
        for (x, y, w, h) in proposals:
            iou = calculate_iou(gt_box, (x, y, x + w, y + h))
            if iou > max_iou:
                max_iou = iou  # Update if a higher IoU is found
        # Only count if the maximum IoU exceeds the threshold
        if max_iou >= iou_threshold:
            recalled_boxes += 1
    recall = recalled_boxes / len(ground_truth_boxes)
    return recall

# Function to evaluate proposals on a single image using ABO
def calc_abo(proposals, ground_truth_boxes):
    sum_max_ious = 0
    for gt_box in ground_truth_boxes:
        max_iou = 0
        for (x, y, w, h) in proposals:
            iou = calculate_iou(gt_box, (x, y, x + w, y + h))
            max_iou= max(max_iou, iou)

        sum_max_ious += max_iou
    
    abo = sum_max_ious / len(ground_truth_boxes) 
    return abo
    




import re 
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
# Function to extract number from the filename
def get_id(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def prepare_proposals(images_path, annotations_path, proposals_per_image, iou_threshold=0.5,scale=100, sigma=1.2, min_size=300,image_shape=(400,400),count=None):

    image_height, image_width = image_shape

    # Get image and label paths
    files = os.listdir(images_path)
    image_paths = np.array(list(filter(lambda file: file.endswith(".jpg"), files)))
    label_paths = np.array(list(filter(lambda file: file.endswith(".xml"), files)))

    

    image_paths = sorted(image_paths, key=get_id)
    label_paths = sorted(label_paths, key=get_id)

   
    
    if count != None:
        image_paths = image_paths[:count]
        label_paths = label_paths[:count]
        
    proposal_data = np.zeros((len(image_paths), proposals_per_image, 4))
    labels = np.zeros((len(image_paths), proposals_per_image, 1))  # Array for labels
    
    # Function to process a single image
    def process_image(filename):
        id = get_id(filename)
        image_path = os.path.join(images_path, filename)
        annotation_file = os.path.join(annotations_path, filename.replace('.jpg', '.xml'))

        image = cv2.imread(image_path)
        original_size = image.shape[:2]
        image = cv2.resize(image, image_shape)
        ground_truth_boxes = parse_xml(annotation_file)
        # scale ground truth boxes
        original_height, original_width = original_size
        height_ratio = image_height / original_height
        width_ratio = image_width / original_width
        
        ground_truth_boxes = [
            (
                int(xmin * width_ratio),
                int(ymin * height_ratio),
                int(xmax * width_ratio),
                int(ymax * height_ratio)
            )
            for (xmin, ymin, xmax, ymax) in ground_truth_boxes
        ]
                
        proposals = get_proposals(image, proposals_per_image, scale=scale, sigma=sigma, min_size=min_size)
        
        
        image_proposals = np.zeros((proposals_per_image, 4))
        image_labels = np.zeros((proposals_per_image, 1))

        for i, (x, y, w, h) in enumerate(proposals):
            proposal_box = (x, y, x + w, y + h)
            max_iou = 0  # Initialize maximum IoU for the current proposal

            for gt_box in ground_truth_boxes:
                iou = calculate_iou(proposal_box, gt_box)
                if iou > max_iou:
                    max_iou = iou  # Update maximum IoU if a new higher value is found

            # Assign label based on maximum IoU found
            label = 1 if max_iou >= iou_threshold else 0

            image_proposals[i, :] = [x, y, w, h]
            image_labels[i] = label

        return id, image_proposals, image_labels

    # Use ThreadPoolExecutor to parallelize the image processing
    with ThreadPoolExecutor(max_workers= 8) as executor:
        futures = {executor.submit(process_image, filename): filename for filename in image_paths}
        for future in futures:
            try:
                id, image_proposals, image_labels = future.result()
                proposal_data[id - 1, :, :] = image_proposals
                labels[id - 1, :] = image_labels
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    return proposal_data, labels


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image