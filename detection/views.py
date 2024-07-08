from django.shortcuts import render
import cv2
import numpy as np
import os

def detect_plate(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        img_np = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        plates = plat_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))
        
        for (x, y, w, h) in plates:
            cv2.putText(img, text='License Plate', org=(x-3, y-3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(0, 0, 255), thickness=1, fontScale=0.6)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            plate_img = img[y:y+h, x:x+w]
            plate_img_path = os.path.join('media', 'plate_image.jpg')
            cv2.imwrite(plate_img_path, plate_img)
        
        img_path = os.path.join('media', 'processed_image.jpg')
        cv2.imwrite(img_path, img)
        
        return render(request, 'result.html', {'image_path': img_path, 'plate_image_path': plate_img_path})
    
    return render(request, 'upload.html')
