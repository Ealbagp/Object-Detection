# %%
import os
import cv2
import selectivesearch
import sys

sys.path.append("..")  # Go up one level, adjust as necessary

from module.utils import  (
    parse_xml, prepare_proposals,get_proposals, calculate_iou, load_image, get_id,calc_recall, calc_abo
    , calculate_proposal_label, resize_box, resize_boxes
)
import matplotlib.pyplot as plt
import numpy as np
img_path = "../Potholes/annotated-images/"
anno_path = "../Potholes/annotated-images/"
proposal_path = "tmp/"
# This does not scale. We should save annotation proposals in a file. 


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

img_files = os.listdir(img_path)
proposal_files = os.listdir(proposal_path)

image_paths = np.array(list(filter(lambda file: file.endswith(".jpg"), img_files)))
label_paths = np.array(list(filter(lambda file: file.endswith(".xml"), img_files)))
proposal_paths = np.array(list(filter(lambda file: file.endswith(".xml"), proposal_files)))
# sort the files
image_paths = sorted(image_paths, key=get_id)
label_paths = sorted(label_paths, key=get_id)
proposal_paths = sorted(proposal_files, key=get_id)
print(image_paths)

# Limit to the first 100 images
image_paths = image_paths[:100]
label_paths = label_paths[:100]
proposal_paths = proposal_paths[:100]

gt_boxes = [parse_xml(anno_path + label_path) for label_path in label_paths]
images = [load_image(img_path + img) for img in image_paths]
proposals = [parse_xml(proposal_path + proposal) for proposal in proposal_paths]
gt_boxes = [resize_boxes(boxs, (image.shape[1], image.shape[0]), IMAGE_SIZE) for boxs, image in zip(gt_boxes, images)]

images = [cv2.resize(image, IMAGE_SIZE) for image in images]


# resize ground truth to ensure that they match the image size


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
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                
            # cv2.putText(image, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
image_count = 10
def label_proposals(proposals, ground_truth_boxes, iou_threshold=0.5, scale_x=1.0, scale_y=1.0):
    labeled_proposals = []

    # ground truth need to be scaled
    
    for proposal_set, gt_boxes in zip(proposals, ground_truth_boxes):
        image_labeled_proposals = []
        for proposal in proposal_set:
            x,y,w,h = proposal
            x = x * scale_x
            y = y * scale_y
            w = w * scale_x
            h = h * scale_y
            proposal = (x, y, w, h)
            label = calculate_proposal_label(proposal, gt_boxes, iou_threshold)
            
            image_labeled_proposals.append(label)
        labeled_proposals.append(image_labeled_proposals)

    return labeled_proposals


labels = label_proposals(proposals=proposals, ground_truth_boxes=gt_boxes, iou_threshold=0.5)

print(proposals[0])

for i in range(image_count):
    # Calculate scale_x and scale_y
    # scale_x = images[i].shape[1] / IMAGE_WIDTH
    # scale_y = images[i].shape[0] / IMAGE_HEIGHT 
    # convert to xywh for proposals

    visualize_image(images[i], gt_boxes[i], labels[i], proposals[i])



# Example usage:
# proposals = [[(x, y, w, h), ...], ...]
# ground_truth_boxes = [[(xmin, ymin, xmax, ymax), ...], ...]
# labeled_proposals = label_proposals(proposals, ground_truth_boxes)
# %%
