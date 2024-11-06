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
def get_proposals(image, num_proposals):
    _, regions = selectivesearch.selective_search(image, scale=10, sigma=0.8, min_size=100)
    
    filtered_regions = list(filter(lambda r:  r['rect'][2] > 20 and r['rect'][3] > 20, regions))
    
    # Ensure the number of proposals does not exceed available regions
    num_proposals = min(num_proposals, len(filtered_regions))
    
    # Randomly select unique indexes from the filtered regions
    selected_regions = random.sample(filtered_regions, num_proposals)
    
    # Extract and return the (x, y, w, h) proposals
    proposals = [(r['rect'][0], r['rect'][1], r['rect'][2], r['rect'][3]) for r in selected_regions]
    
    return proposals

# Function to calculate the IoU between to proposals(two boxes)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_Area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_Area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxA_Area + boxB_Area - interArea)
    return iou




import re 
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
# Function to extract number from the filename
def get_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def prepare_proposals(images_path, annotations_path, proposals_per_image, iou_threshold=0.5, count=None):

    # Get image and label paths
    files = os.listdir(images_path)
    image_paths = np.array(list(filter(lambda file: file.endswith(".jpg"), files)))
    label_paths = np.array(list(filter(lambda file: file.endswith(".xml"), files)))

    proposal_data = np.zeros((len(image_paths), proposals_per_image, 4))
    labels = np.zeros((len(image_paths), proposals_per_image, 1))  # Array for labels

    image_paths = sorted(image_paths, key=get_number)
    label_paths = sorted(label_paths, key=get_number)

    if count != None:
        image_paths = image_paths[:count]
        label_paths = label_paths[:count]
    
    # Function to process a single image
    def process_image(filename):
        id = get_number(filename)
        image_path = os.path.join(images_path, filename)
        annotation_file = os.path.join(annotations_path, filename.replace('.jpg', '.xml'))

        image = cv2.imread(image_path)
        ground_truth_boxes = parse_xml(annotation_file)
        proposals = get_proposals(image, proposals_per_image)

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