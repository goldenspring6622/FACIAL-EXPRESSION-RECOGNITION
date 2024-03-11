import cv2
import argparse
import time
import numpy as np
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

try:
    emotions = {
        0: "Angry",
        1: "Sad",
        2: "Fear",
        3: "Happy",
        4: "Neutral",
        5: "Disgust",
        6: "Surprise",
    }
    mp_face_detection = mp.solutions.face_detection

    classes = np.array(list(map(lambda x: emotions[x], emotions)))

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(197, 197),
            # tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),
        ]
    )

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Emotion detection from video")
    parser.add_argument(
        "--model_path", type=str, help="Path to the model file (HDF5 format)"
    )
    parser.add_argument(
        "--img_path",
        default=None,
        type=str,
        help="Path to the input image for classification",
    )

    parser.add_argument("--video_path", type=str, help="Path to the input video file")

    args = parser.parse_args()
    print(args)
    # Use the provided video_path or use a default path if not provided
    img_path = args.img_path
    video_path = args.video_path
    model_path = args.model_path if args.model_path else "models/model.h5"
    model = load_model(model_path, compile=False)
    if img_path:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        ) as face_detector:
            frame = cv2.imread(img_path)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_detector.process(rgb_frame)
            frame_height, frame_width, c = frame.shape
            if results.detections:
                for face in results.detections:
                    face_react = np.multiply(
                        [
                            face.location_data.relative_bounding_box.xmin,
                            face.location_data.relative_bounding_box.ymin,
                            face.location_data.relative_bounding_box.width,
                            face.location_data.relative_bounding_box.height,
                        ],
                        [frame_width, frame_height, frame_width, frame_height],
                    ).astype(int)

                    x, y, w, h = face_react
                    boxs = [x, y, x + w, y + h]
                    # gray_image = cv2.resize(frame, (48, 48))
                    # gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    # print(gray_image.shape)
                    # If you want the resulting array to be of integer type (e.g., uint8)
                    roi_gray = frame[y : y + h, x : x + w]
                    img_array = roi_gray.astype("float") / 255.0
                    img_array = image.img_to_array(img_array)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = data_augmentation(img_array)
                    # # Get model predictions
                    predictions = model.predict(img_array)
                    # Interpret the predictions
                    predicted_class = np.argmax(predictions)
                    print(classes[predicted_class])
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
                    cv2.putText(
                        frame,
                        str(classes[predicted_class]),
                        (boxs[0], boxs[1]),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (100, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("frame", frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:
        if not video_path:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        ) as face_detector:
            frame_counter = 0
            fonts = cv2.FONT_HERSHEY_PLAIN
            start_time = time.time()

            while True:
                try:
                    frame_counter += 1
                    ret, frame = cap.read()
                    if ret is False:
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = face_detector.process(rgb_frame)
                    frame_height, frame_width, c = frame.shape
                    if results.detections:
                        for face in results.detections:
                            face_react = np.multiply(
                                [
                                    face.location_data.relative_bounding_box.xmin,
                                    face.location_data.relative_bounding_box.ymin,
                                    face.location_data.relative_bounding_box.width,
                                    face.location_data.relative_bounding_box.height,
                                ],
                                [frame_width, frame_height, frame_width, frame_height],
                            ).astype(int)

                            x, y, w, h = face_react
                            boxs = [x, y, x + w, y + h]
                            # gray_image = cv2.resize(frame, (48, 48))
                            # gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                            # print(gray_image.shape)
                            # If you want the resulting array to be of integer type (e.g., uint8)
                            roi_gray = frame[y : y + h, x : x + w]

                            img_array = roi_gray.astype("float") / 255.0
                            img_array = image.img_to_array(img_array)
                            img_array = np.expand_dims(img_array, axis=0)
                            img_array = data_augmentation(img_array)

                            # # Get model predictions
                            predictions = model.predict(img_array)
                            # Interpret the predictions
                            predicted_class = np.argmax(predictions)
                            print(classes[predicted_class])
                            cv2.putText(
                                frame,
                                str(classes[predicted_class]),
                                (boxs[0], boxs[1]),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1,
                                (100, 255, 0),
                                3,
                                cv2.LINE_AA,
                            )

                            frame = cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (0, 0, 255), 6
                            )

                    fps = frame_counter / (time.time() - start_time)
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.2f}",
                        (30, 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (100, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
                except Exception as e:
                    print(str(e))
            cap.release()
            cv2.destroyAllWindows()
except Exception as e:
    print(str(e))
