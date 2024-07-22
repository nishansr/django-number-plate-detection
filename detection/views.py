from django.shortcuts import render
import cv2
import numpy as np
import os

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
        
        return render(request, 'result.html', {
            'image_path': img_path,
            'plate_image_path': plate_img_path,
            'global_thresh_path': global_thresh_path,
            'adaptive_thresh_path': adaptive_thresh_path,
            'otsu_thresh_path': otsu_thresh_path,
            'median_filtered_path': median_filtered_path,
            'prewitt_filtered_path': prewitt_filtered_path,
            'sobel_filtered_path': sobel_filtered_path,
            'laplacian_filtered_path': laplacian_filtered_path
        })
    
    return render(request, 'upload.html')
