uploading file

scp path/to/Potholes.zip s242064@login1.gbar.dtu.dk:~/

^ to home directory

# Project Update

## Overview
This project involves the implementation and testing of different object proposal generation methods for pothole detection. The current status and functionality of each component are detailed below.

---

## Current Components

### 1. `data_visualization`
- **Description**: This folder contains scripts to visualize the dataset images along with the ground truth bounding boxes (labels).
- **Purpose**: Provides an initial understanding of the dataset by displaying the images with their corresponding ground truth labels, helping to verify the accuracy of annotations.

### 2. `edges_boxes`
- **Description**: This folder implements the Edge Boxes method for generating object proposal boxes.
- **Status**: Currently displays proposal boxes overlaid on images. We need to further tune the Edge Boxes parameters (e.g., `alpha`, `beta`, `maxBoxes`, `minScore`) to better capture the desired regions (i.e., potholes).
- **Next Steps**: Experiment with different parameter settings to obtain proposals that focus on larger potholes without including excessive small details.

### 3. `selective_search`
- **Description**: This folder contains the Selective Search (SS) method for generating object proposal boxes.
- **Status**: Displays proposal boxes on images. Similar to Edge Boxes, parameter tuning is needed for Selective Search to optimize the proposals.
- **Next Steps**: Adjust parameters (e.g., `scale`, `sigma`, `min_size`) to improve proposal quality for pothole detection.

### 4. `model`
- **Description**: Contains the Edge Boxes model file (`model.yml.gz`) that was sourced online and used to implement Edge Boxes proposals.
- **Purpose**: This model file is necessary for Edge Boxes to function and is part of the Structured Edge Detection model, required to generate edges and orientations.

### 5. `Evaluation_of_Proposals_for_SS`
- **Description**: Contains the metrics Recall and ABO for analizing the proposals. And a implementation loop to evaluate them using the selective_search method.
- **Purpose**: This model file is necessary evaluates the proposals generated and is intended to solve Task 3.

### 5. `Save and Label Proposals for Training`
- **Description**: Contains a function to generate the labels for proposals according to whether the porposal overlaps with a ground truth pothole or not.
- **Purpose**: This is neccesary for the creation of a DataLoader in the next task.


---

## To-Do List

1. **Implement Metrics** (DONE for selective search, there is left to change the loop for the Edge Boxes)
   - **Recall**: Calculate recall for each method (Edge Boxes and Selective Search) to measure how well the proposals cover the ground truth potholes.
   - **Mean Average Best Overlap (MABO)**: Implement the MABO metric to evaluate the average quality of proposals across images.
   
2. **Save and Label Proposals for Training** (DONE for selective search, there is left to change it for the Edge Boxes).
   - **Save Proposals**: Save the generated proposals from both Edge Boxes and Selective Search methods.
   - **Label Proposals**: Prepare the proposals for object detector training by assigning a label to each proposal. This includes:
     - Assigning a class label if the proposal overlaps with a ground truth pothole.
     - Assigning a background label if the proposal does not overlap with any ground truth box.

---

## Next Steps and Team Collaboration

To proceed with the project, team members can start focusing on:
- Experimenting with parameter tuning for both Edge Boxes and Selective Search to optimize proposal quality.
- Implementing the recall and MABO metrics to evaluate the effectiveness of the proposal generation methods.
- Preparing and organizing the proposals for the training dataset, ensuring all proposals have accurate labels (class or background).

Feel free to check each folder for specific implementations, and reach out with any questions or suggestions for further improvements.

