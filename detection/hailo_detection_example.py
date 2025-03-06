import gi
import numpy as np
import cv2
import hailo
from gi.repository import Gst, GLib

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

class TwoClassDetector(app_callback_class):
    def __init__(self):
        super().__init__()
        self.class_mapping = {
            "bolt": "bolt",
            "nut": "nut",
            "screw_body": "bolt",
            "screw_head": "bolt"
        }

    def map_classes(self, label):
        return self.class_mapping.get(label, "unknown")

# Callback function to process detections
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    format, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        mapped_label = user_data.map_classes(label)
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if mapped_label in ["bolt", "nut"]:
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {mapped_label} Confidence: {confidence:.2f}\n")
            detection_count += 1
    
    if user_data.use_frame:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = TwoClassDetector()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
