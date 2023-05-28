import torch
import numpy as np
import cv2
import pyttsx3
from time import time
from ultralytics import YOLO
from collections import defaultdict

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator  # , MaskAnnotator


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(
            color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
        # self.mask_annotator = MaskAnnotator(color=ColorPalette())

    def load_model(self):

        # load a pretrained YOLOv8n model
        model = YOLO("D:\python\hackathon/best.pt")
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame)

        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        detected_objects = defaultdict(lambda: 0)

        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:

                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        # Setup detections for visualization
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            # mask=results[0].masks.data,
        )
        for (_, _, class_id, _) in detections:
            detected_objects[self.CLASS_NAMES_DICT[class_id]] += 1

        print(dict(detected_objects))

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=self.labels)
        # frame = self.mask_annotator.annotate(frame=frame, detections=detections)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture("http://25.94.35.235:8080/video")

        while True:

            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(20) & 0xFF == ord('f'):
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=0)
detector()
