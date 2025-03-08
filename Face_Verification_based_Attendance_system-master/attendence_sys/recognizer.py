import face_recognition
import numpy as np
import cv2
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def Recognizer(details):
    video = cv2.VideoCapture(0)

    known_face_encodings = []
    known_face_names = []

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_dir = os.path.join(base_dir, "static", "images", "Student_Images",
                             details['branch'], details['year'], details['section'])
    logger.debug(f"Image directory: {image_dir}")

    names = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('jpg', 'png')):
                path = os.path.join(root, file)
                img = face_recognition.load_image_file(path)
                label = file.rsplit('.', 1)[0]
                try:
                    img_encoding = face_recognition.face_encodings(img)[0]
                    known_face_names.append(label)
                    known_face_encodings.append(img_encoding)
                except IndexError:
                    logger.warning(f"No face found in image: {path}. Skipping.")

    logger.debug("Starting face recognition...")
    done_message_displayed = False

    cv2.namedWindow("Face Recognition Panel", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Face Recognition Panel", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = video.read()
        if not ret:
            logger.error("Failed to capture frame. Exiting.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)
                if name not in names:
                    names.append(name)
            else:
                face_names.append("Unknown")

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        if done_message_displayed:
            height, width, _ = frame.shape
            cv2.putText(frame, "Done!", (int(width / 3), int(height / 4)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Image saved successfully!", (int(width / 6), int(height / 3)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Face Recognition Panel", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            save_dir = os.path.abspath("captured_images")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, "captured_image.jpg")
            success = cv2.imwrite(save_path, frame)
            if success:
                logger.info(f"Image saved successfully at {save_path}")
                done_message_displayed = True
            else:
                logger.error("Failed to save the image.")

        if key == ord('q'):
            logger.info("Exiting.")
            break

    video.release()
    cv2.destroyAllWindows()
    logger.info(f"Recognized names: {names}")
    return names
