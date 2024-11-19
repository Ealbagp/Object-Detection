import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model')

# Load your data
data = ...  # Replace with your data loading/preprocessing

# Predict bounding boxes and scores
predictions = model.predict(data)

# Example format of predictions: [boxes, scores]
# - boxes: array of shape (num_boxes, 4), where each box is [x_min, y_min, x_max, y_max]
# - scores: array of shape (num_boxes,), confidence scores for each box
boxes, scores = predictions['boxes'], predictions['scores']  # Adjust based on your model's output format

# Perform Non-Maximum Suppression
iou_threshold = 0.5  # IoU threshold to suppress overlapping boxes
score_threshold = 0.3  # Minimum score to keep a box

selected_indices = tf.image.non_max_suppression(
    boxes=boxes,
    scores=scores,
    max_output_size=100,  # Maximum number of boxes to keep
    iou_threshold=iou_threshold,
    score_threshold=score_threshold
)

# Get the filtered boxes and scores
filtered_boxes = tf.gather(boxes, selected_indices).numpy()
filtered_scores = tf.gather(scores, selected_indices).numpy()

# Print results
for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
    print(f"Box {i}: {box}, Score: {score}")
