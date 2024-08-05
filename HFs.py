import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

facedetection = mp.solutions.face_detection
mpdrawing = mp.solutions.drawing_utils

with facedetection.FaceDetection(model_selection =0 , min_detection_confidence=0.4 ) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print ("error")
            continue 
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)    
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)             
                face_region = image[y:y+h, x:x+w]           
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  
                image[y:y+h, x:x+w] = blurred_face
                
                
        cv2.imshow("cam" , image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()