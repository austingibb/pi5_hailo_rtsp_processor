import sys
import argparse
import yaml
import os
import cv2
from threading import Thread, Lock
import numpy as np
import hailo
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Hailo helpers
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# If you have your custom RTSP Pose app:
# from my_subclass_file import MyCustomRTSPPoseEstimationApp

##############################################
# 1. Minimal parser for -config
##############################################
def parse_local_args():
    local_parser = argparse.ArgumentParser(add_help=False)
    local_parser.add_argument('-config', type=str, default='config.yaml',
                              help='Path to the YAML configuration file')
    known_args, leftover = local_parser.parse_known_args()
    return known_args, leftover

##############################################
# 2. Callback + Keypoints (same as before)
##############################################
def get_keypoints():
    return {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }

KEYPOINTS_DICT = get_keypoints()

class UserAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.lock = Lock()
        self.latest_frame = None

    def set_frame(self, frame):
        with self.lock:
            self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame_number = user_data.get_count()

    fmt, width, height = get_caps_from_pad(pad)
    if not user_data.use_frame or (fmt is None):
        return Gst.PadProbeReturn.OK

    frame = get_numpy_from_buffer(buffer, fmt, width, height)
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    for detection in detections:
        detection_count += 1
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}",
                    (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) > 0:
            pts = landmarks[0].get_points()
            for name, idx in KEYPOINTS_DICT.items():
                point = pts[idx]
                px = int((point.x() * bbox.width()) + bbox.xmin())
                py = int((point.y() * bbox.height()) + bbox.ymin())
                cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

    cv2.putText(frame, f"Frame #{frame_number} / Detections: {detection_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    user_data.set_frame(frame_bgr)
    return Gst.PadProbeReturn.OK

##############################################
# 3. Main
##############################################
def main():
    # -----------------------------------------
    # A) Parse our local -config argument
    # -----------------------------------------
    known_args, leftover = parse_local_args()
    config_path = known_args.config

    # Load YAML
    if not os.path.exists(config_path):
        print(f"Error: config file not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Just for debugging:
    print("Local leftover before strip:", leftover)

    # -----------------------------------------
    # B) Strip these arguments from sys.argv
    #    so Hailo’s parser won't see -config
    # -----------------------------------------
    import sys
    # Rebuild sys.argv with leftover only
    sys.argv = [sys.argv[0]] + leftover

    # Now any subsequent imports or calls to
    # Hailo code that do argparse will not see
    # -config in sys.argv.
    print("sys.argv after strip:", sys.argv)

    # -----------------------------------------
    # C) Use your config data for RTSP, mode, etc.
    # -----------------------------------------
    mode = config.get('mode', 'object')
    username = config.get('username', '')
    password = config.get('password', '')
    camera_ip = config.get('camera_ip', '')
    port = config.get('port', 554)
    stream_path = config.get('stream_path', '')
    resize_pct = config.get('resize_percentage', 50)
    hailo_model_path = config.get('model_path', 'your_model.hailo')

    if stream_path:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}/{stream_path}'
    else:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}'

    # Example pipeline for object detection
    pipeline = (
        f"rtspsrc latency=200 location={rtsp_url} ! "
        "queue ! decodebin ! videoconvert ! video/x-raw,format=RGB ! "
        f"hailonet name=net model_path={hailo_model_path} ! "
        "hailofilter name=filter ! "
        "appsink emit-signals=True name=sink max-buffers=1 drop=true"
    )

    user_data = UserAppCallback()
    user_data.use_frame = True

    # -----------------------------------------
    # D) Create your app depending on mode
    # -----------------------------------------
    if mode.lower() == "pose":
        # e.g. If you have a custom RTSP Pose class
        # app = MyCustomRTSPPoseEstimationApp(app_callback, user_data)
        # app.video_source = rtsp_url
        # app.hef_path = hailo_model_path
        # etc...
        #
        # Or if you have a direct constructor that takes pipeline
        # (But GStreamerPoseEstimationApp normally doesn't).
        #
        # For demonstration, let's pretend we have GStreamerPoseEstimationApp
        # that can handle a pipeline. If it can't, you'd do the subclass approach.
        app = GStreamerPoseEstimationApp(app_callback, user_data)  # no pipeline param
        # Then set app.video_source or patch the pipeline in your custom version.
        #
        # This is pseudo-code:
        # app.video_source = rtsp_url
        # app.hef_path = hailo_model_path
        # ...
    else:
        # Object detection approach
        app = GStreamerDetectionApp(app_callback, user_data, pipeline)

    # -----------------------------------------
    # E) Launch and show frames
    # -----------------------------------------
    gst_thread = Thread(target=app.run)
    gst_thread.start()

    cv2.namedWindow("Hailo Stream", cv2.WINDOW_NORMAL)
    while True:
        frame_bgr = user_data.get_frame()
        if frame_bgr is not None:
            h, w = frame_bgr.shape[:2]
            new_w = int(w * (resize_pct / 100.0))
            new_h = int(h * (resize_pct / 100.0))
            resized_frame = cv2.resize(frame_bgr, (new_w, new_h))
            cv2.imshow("Hailo Stream", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            app.stop()
            break

    gst_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
