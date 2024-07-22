from django.shortcuts import render
import cv2
import numpy as np
import os

def get_chain_code(img, directions=4):
    # Convert image to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chain_code = []
    if directions == 4:
        direction_map = {
            (0, 1): 0,
            (-1, 0): 1,
            (0, -1): 2,
            (1, 0): 3
        }
    else:  # 8 directions
        direction_map = {
            (0, 1): 0,
            (-1, 1): 1,
            (-1, 0): 2,
            (-1, -1): 3,
            (0, -1): 4,
            (1, -1): 5,
            (1, 0): 6,
            (1, 1): 7
        }

    if contours:
        contour = contours[0]
        prev_point = contour[0][0]
        for point in contour[1:]:
            diff = (point[0][1] - prev_point[1], point[0][0] - prev_point[0])
            if diff in direction_map:
                chain_code.append(direction_map[diff])
            prev_point = point[0]

    return chain_code

def get_first_difference(chain_code, directions=4):
    first_diff = []
    for i in range(1, len(chain_code)):
        first_diff.append((chain_code[i] - chain_code[i - 1]) % directions)
    return first_diff


def detect_plate(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        img_np = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Global thresholding
        _, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        
        # Otsu's thresholding
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Median filter
        median_filtered = cv2.medianBlur(gray, 5)
        
        # Prewitt filter (approximation using Sobel)
        prewitt_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        prewitt_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        prewitt_filtered = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
        
        # Sobel filter
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_filtered = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        
        # Laplacian filter
        laplacian_filtered = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Save thresholded and filtered images
        global_thresh_path = os.path.join('media', 'global_thresh.jpg')
        adaptive_thresh_path = os.path.join('media', 'adaptive_thresh.jpg')
        otsu_thresh_path = os.path.join('media', 'otsu_thresh.jpg')
        median_filtered_path = os.path.join('media', 'median_filtered.jpg')
        prewitt_filtered_path = os.path.join('media', 'prewitt_filtered.jpg')
        sobel_filtered_path = os.path.join('media', 'sobel_filtered.jpg')
        laplacian_filtered_path = os.path.join('media', 'laplacian_filtered.jpg')
        
        cv2.imwrite(global_thresh_path, global_thresh)
        cv2.imwrite(adaptive_thresh_path, adaptive_thresh)
        cv2.imwrite(otsu_thresh_path, otsu_thresh)
        cv2.imwrite(median_filtered_path, median_filtered)
        cv2.imwrite(prewitt_filtered_path, prewitt_filtered)
        cv2.imwrite(sobel_filtered_path, sobel_filtered)
        cv2.imwrite(laplacian_filtered_path, laplacian_filtered)

        # Detect plates
        plate_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        plates = plate_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))
        
        plate_img_path = None
        for (x, y, w, h) in plates:
            cv2.putText(img, text='License Plate', org=(x-3, y-3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(0, 0, 255), thickness=1, fontScale=0.6)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            plate_img = img[y:y+h, x:x+w]
            plate_img_path = os.path.join('media', 'plate_image.jpg')
            cv2.imwrite(plate_img_path, plate_img)
        
        img_path = os.path.join('media', 'processed_image.jpg')
        cv2.imwrite(img_path, img)
        
        chain_code_4 = get_chain_code(gray, directions=4)
        first_diff_4 = get_first_difference(chain_code_4, directions=4)

        chain_code_8 = get_chain_code(gray, directions=8)
        first_diff_8 = get_first_difference(chain_code_8, directions=8)

        return render(request, 'result.html', {
            'image_path': img_path,
            'plate_image_path': plate_img_path,
            'global_thresh_path': global_thresh_path,
            'adaptive_thresh_path': adaptive_thresh_path,
            'otsu_thresh_path': otsu_thresh_path,
            'median_filtered_path': median_filtered_path,
            'prewitt_filtered_path': prewitt_filtered_path,
            'sobel_filtered_path': sobel_filtered_path,
            'laplacian_filtered_path': laplacian_filtered_path,
            'chain_code_4': chain_code_4,
            'first_diff_4': first_diff_4,
            'chain_code_8': chain_code_8,
            'first_diff_8': first_diff_8
        })
    
    return render(request, 'upload.html')
