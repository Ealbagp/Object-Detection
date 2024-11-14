#%%
import os
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import random
import tqdm
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


import cv2
import random

# Function to run Selective Search and obtain proposals using OpenCV
def get_proposals(image, num_proposals, scale=25, sigma=0.8, min_size=300):
    # Initialize OpenCV's selective search segmentation object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    # Set the base image
    ss.setBaseImage(image)
    
    # Create a graph segmentation based on the provided scale, sigma, and min_size
    gs = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=sigma, k=scale, min_size=min_size)
    ss.addGraphSegmentation(gs)
    
    # Create selective search strategies
    color_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    texture_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    size_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    fill_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    
    # Combine the strategies into one
    strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(
        color_strategy, texture_strategy, size_strategy, fill_strategy)
    ss.addStrategy(strategy)
    ss.switchToSelectiveSearchQuality()
    # Process the image to get the proposed regions
    rects = ss.process()
    
    # Filter regions based on size
    filtered_regions = []
    for x, y, w, h in rects:
        # if w > 20 and h > 20:
        filtered_regions.append((x, y, w, h))
    
    # Ensure the number of proposals does not exceed available regions
    num_proposals = min(num_proposals, len(filtered_regions))
    
    # Randomly select unique indexes from the filtered regions
    selected_regions = random.sample(filtered_regions, num_proposals)
    
    return selected_regions  # Returns a list of (x, y, w, h) tuples

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

# %%

# unit test for calculate_iou
def test_calculate_iou():
    boxA = (0, 0, 10, 10)
    boxB = (5, 5, 15, 15)
    assert calculate_iou(boxA, boxB) == 0.14285714285714285

test_calculate_iou()

# %%

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
    

def from_min_max_to_xywh(box):
    xmin, ymin, xmax, ymax = box
    return (xmin, ymin, xmax - xmin, ymax - ymin)  

def from_xywh_to_min_max(box):
    xmin, ymin , w, h = box
    return (xmin, ymin, xmin + w, ymin + h)


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
        
    proposal_data_dict = {}
    labels_dict = {}
    
    # Function to process a single image
    def process_image(filename,scale, sigma):
        id = get_id(filename)
        image_path = os.path.join(images_path, filename)
        annotation_file = os.path.join(annotations_path, filename.replace('.jpg', '.xml'))

        image = cv2.imread(image_path)
        original_size = image.shape[:2]
        image = cv2.resize(image, image_shape)
        ground_truth_boxes = parse_xml(annotation_file)
        # scale ground truth boxes
        original_height, original_width = original_size
        
        ground_truth_boxes = [resize_box(box, (original_width, original_height), image_shape) for box in ground_truth_boxes]
        
        
                
        proposals = get_proposals(image, proposals_per_image, scale=scale, sigma=sigma, min_size=min_size)
        
        
        image_proposals = []
        image_labels = []

        for (x, y, w, h) in proposals:
            proposal_box = from_xywh_to_min_max((x, y, w, h))
            max_iou = 0  # Initialize maximum IoU for the current proposal

            for gt_box in ground_truth_boxes:
                iou = calculate_iou(proposal_box, gt_box)
                if iou > max_iou:
                    max_iou = iou  # Update maximum IoU if a new higher value is found

            label = 1 if max_iou >= iou_threshold else 0

            image_proposals.append([x, y, w, h])
            image_labels.append([label])

        # Convert lists to numpy arrays
        image_proposals = np.array(image_proposals)
        image_labels = np.array(image_labels)
        
        return id, image_proposals, image_labels
    
    cpu_count = os.cpu_count()*0.8
    cpu_count = min(cpu_count, len(image_paths))
    
    # Use ThreadPoolExecutor to parallelize the image processing
    with ThreadPoolExecutor(max_workers= cpu_count) as executor:
        futures = {executor.submit(process_image, filename, scale, sigma): filename for filename in image_paths}
        for future in tqdm.tqdm(futures):
            
            id, image_proposals, image_labels = future.result()
            proposal_data_dict[id] = image_proposals
            labels_dict[id] = image_labels
           
    proposal_data = [proposal_data_dict[id] for id in sorted(proposal_data_dict.keys())]
    labels = [labels_dict[id] for id in sorted(labels_dict.keys())] 
    
    return proposal_data, labels


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def visualize_image(image, boxes,labels, proposals=None, scale_x=1.0, scale_y=1.0):
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
            x = x * scale_x
            y = y * scale_y
            w = w * scale_x
            h = h * scale_y

            x, y, w, h = int(x), int(y), int(w), int(h)
            if label == 1:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                
            # cv2.putText(image, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def visualize_image(image, boxes, proposals=None, scale_x=1.0, scale_y=1.0):
    # Adjust ground truth boxes according to the scale
    adjusted_boxes = [(int(xmin * scale_x), int(ymin * scale_y), int(xmax * scale_x), int(ymax * scale_y)) for xmin, ymin, xmax, ymax in boxes]
    
    # Convert color for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw ground truth boxes in blue
    for (xmin, ymin, xmax, ymax) in adjusted_boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    # Draw Selective Search proposals in green if provided
    if proposals is not None:
        for (x, y, w, h) in proposals:
            x, y, w, h = int(x), int(y), int(w), int(h)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.putText(image, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

def calculate_proposal_label(proposal, ground_truth_boxes, iou_threshold=0.5):
    proposal_box = from_xywh_to_min_max(proposal)
    max_iou = 0  # Initialize maximum IoU for the current proposal

    for gt_box in ground_truth_boxes:
        iou = calculate_iou(proposal_box, gt_box)
        if iou > max_iou:
            max_iou = iou  # Update maximum IoU if a new higher value is found

    label = 1 if max_iou >= iou_threshold else 0
    return label

# Resize box
def resize_box(box, original_size, new_size):
    original_width, original_height = original_size
    new_width, new_height = new_size

    height_ratio = new_height / original_height
    width_ratio = new_width / original_width
    
    
    
    
    

    xmin, ymin, xmax, ymax = box
    new_xmin = int(xmin * width_ratio)
    new_ymin = int(ymin * height_ratio)
    new_xmax = int(xmax * width_ratio)
    new_ymax = int(ymax * height_ratio)

    return new_xmin, new_ymin, new_xmax, new_ymax

def resize_boxes(boxes, original_size, new_size):
    return [resize_box(box, original_size, new_size) for box in boxes]