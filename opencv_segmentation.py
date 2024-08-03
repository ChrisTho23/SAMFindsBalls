import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def find_largest_circle_and_closest_object(image_path):
    def find_largest_circle(contours):
        largest_circle = None
        max_radius = 0
        largest_index = -1
        for index, contour in enumerate(contours):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
            if radius > max_radius and circularity > 0.8:
                largest_circle = ((int(x), int(y)), int(radius))
                max_radius = radius
                largest_index = index
        return largest_circle, largest_index

    def find_closest_object(bounding_boxes, target_center):
        closest = None
        min_distance = float('inf')
        for box in bounding_boxes:
            box_center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
            dist = euclidean(box_center, target_center)
            if dist < min_distance:
                min_distance = dist
                closest = box
        return closest

    # Read and process the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest circle
    largest_circle, index = find_largest_circle(contours)
    
    if largest_circle:
        # Remove the largest circle from contours
        contours = [contour for i, contour in enumerate(contours) if i != index]
        
        # Find bounding boxes for remaining contours
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        
        # Find the closest object
        closest_object = find_closest_object(bounding_boxes, largest_circle[0])
        
        return largest_circle, closest_object
    else:
        return None, None

# Usage
image_path = 'data/cups_initial.png'
largest_circle, closest_object = find_largest_circle_and_closest_object(image_path)

if largest_circle and closest_object:
    # Visualization (optional)
    img = cv2.imread(image_path)
    result = img.copy()
    
    # Draw the largest circle
    cv2.circle(result, largest_circle[0], largest_circle[1], (0, 0, 255), 2)
    
    # Draw the closest object
    cv2.rectangle(result, (closest_object[0], closest_object[1]), 
                  (closest_object[0] + closest_object[2], closest_object[1] + closest_object[3]), 
                  (255, 0, 0), 2)
    
    # Draw a line between them
    closest_center = (closest_object[0] + closest_object[2] // 2, 
                      closest_object[1] + closest_object[3] // 2)
    cv2.line(result, largest_circle[0], closest_center, (0, 255, 0), 2)
    
    cv2.imshow('Detected Objects', result)
    cv2.waitKey(0)
    cv2.imwrite('output_largest_circle_and_closest.png', result)
    cv2.destroyAllWindows()
else:
    print("No circular objects detected or no other objects found")