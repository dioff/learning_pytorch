import cv2


image = cv2.imread(r"Object_Count\image2.jpg")
# template = cv2.imread(r"Object_Count\template_new.png", 0)
template = cv2.imread(r"Object_Count\template.jpg", 0)

template = cv2.Canny(template, 50, 200)

image_copy = image.copy()
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

# Apply template Matching.
result = cv2.matchTemplate(image_copy, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Top left x and y coordinates.
x1, y1 = max_loc
# Bottom right x and y coordinates.
x2, y2 = (x1 + w, y1 + h)
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('Result', image)
# Normalize the result for proper grayscale output visualization.
cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1 )
cv2.imshow('Detected point', result)
cv2.waitKey(0)
cv2.imwrite("Object_Count/image_result.jpg", image)
cv2.imwrite("Object_Count/template_result.jpg", result*255.)