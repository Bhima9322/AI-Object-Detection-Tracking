import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    """
    Main function to initialize the camera, YOLOv8 model, and DeepSORT tracker.
    It tracks objects accurately in the backend but displays them only as 'Human' or 'Object'.
    """
    # 1. Initialize the YOLOv8 model
    model = YOLO("yolov8s.pt") 
    class_names = model.names

    # 2. Initialize the DeepSORT tracker
    tracker = DeepSort(max_age=50, n_init=3)

    # 3. Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open the webcam. Please check your camera connection.")
        return

    print("[INFO] Security Camera started successfully. Press the 'Esc' key to exit.")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture video frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 4. Perform object detection
        results = model(frame, stream=True)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # वस्तूंना जास्त चांगल्या प्रकारे ओळखण्यासाठी आपण Confidence 30% (0.3) ठेवला आहे
                if confidence > 0.3:
                    w = x2 - x1
                    h = y2 - y1
                    
                    # 💡 ट्रॅकरला आपण मूळ नावच देणार आहोत (जेणेकरून तो ट्रॅकिंग थांबवणार नाही)
                    original_name = class_names[class_id]
                    detections.append(([x1, y1, w, h], confidence, original_name))

        # 5. Update the tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # 6. Draw bounding boxes on the screen
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            original_category = track.det_class if track.det_class else "Unknown"
            
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # 💡 मुख्य लॉजिक: स्क्रीनवर फक्त 'Human' आणि 'Object' दाखवणे
            if original_category == "person":
                display_name = "Human"
                box_color = (0, 255, 0)      # माणसासाठी हिरवा रंग
            else:
                display_name = "Object"
                box_color = (0, 165, 255)    # इतर वस्तूंसाठी केशरी रंग
                
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Prepare the label text
            label = f"{display_name} ID: {track_id}"
            
            # Draw a solid background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1)
            
            # Draw the text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 7. Render output
        cv2.imshow("Live AI Security Camera", frame)

        # Exit when 'Esc' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            print("[INFO] Exiting the application securely.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()