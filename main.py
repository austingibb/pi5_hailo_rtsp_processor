import cv2
import yaml
import argparse
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from threading import Thread, Lock
import numpy as np
import hailo

# Hailo helper imports
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

#######################################################################
# 1. Helper: keypoints dictionary for COCO Pose
#######################################################################
def get_keypoints():
    """
    Return a dictionary mapping COCO keypoint names to their indices.
    Example keys: 'left_eye', 'right_eye', 'left_ear', ...
    """
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

#######################################################################
# 2. User-defined callback class
#######################################################################
class UserAppCallback(app_callback_class):
    """
    Extend hailo_apps_infra.hailo_rpi_common.app_callback_class
    so we can store frames & custom variables.
    """
    def __init__(self):
        super().__init__()
        self.lock = Lock()
        self.latest_frame = None

    def set_frame(self, frame):
        """Thread-safe setter for the latest overlaid frame."""
        with self.lock:
            self.latest_frame = frame

    def get_frame(self):
        """Thread-safe getter for the latest overlaid frame."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

#######################################################################
# 3. The GStreamer callback function
#######################################################################
def app_callback(pad, info, user_data):
    # Retrieve the GstBuffer
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment our frame counter
    user_data.increment()
    frame_number = user_data.get_count()

    # Get the caps from the pad: format, width, height
    fmt, width, height = get_caps_from_pad(pad)
    if (not user_data.use_frame) or (fmt is None):
        return Gst.PadProbeReturn.OK

    # Convert buffer to a NumPy array (typically in RGB)
    frame = get_numpy_from_buffer(buffer, fmt, width, height)

    # Extract detections from the hailo buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)  # bounding boxes, labels, confidences

    # This list might exist if your pipeline includes pose landmarks
    # e.g. hailo.HAILO_LANDMARKS objects inside each detection
    # We'll show how to draw them if present
    detection_count = 0

    for detection in detections:
        detection_count += 1
        label = detection.get_label()
        bbox = detection.get_bbox()  # (x_min, y_min, x_max, y_max)
        confidence = detection.get_confidence()

        # Draw bounding boxes (for object detection or "person" in pose)
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{label} {confidence:.2f}",
                    (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Check for pose landmarks (if the pipeline is pose-based, or if
        # the detection includes HAILO_LANDMARKS)
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) > 0:
            # Typically there's just one landmarks object
            pts = landmarks[0].get_points()  # keypoints in [0,1] relative coords
            # We'll just demonstrate drawing eyes or all keypoints
            for name, idx in KEYPOINTS_DICT.items():
                # pts[idx] is the normalized location inside the bounding box
                # point.x(), point.y() in range [0,1].
                point = pts[idx]
                # Convert from relative coords to absolute pixel coords
                px = int((point.x() * bbox.width()) + bbox.xmin())
                py = int((point.y() * bbox.height()) + bbox.ymin())
                # Draw the keypoint
                cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)
                # Optionally label it
                # cv2.putText(frame, name, (px, py),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # Overlay some debug text
    cv2.putText(frame, f"Frame #{frame_number} / Detections: {detection_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Convert to BGR for display in OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    user_data.set_frame(frame_bgr)

    return Gst.PadProbeReturn.OK

#######################################################################
# 4. Main function: parse config, pick pipeline, show frames
#######################################################################
def main():
    # ----------------------------------------------------------
    # Parse CLI args
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(description='Hailo + OpenCV: Object or Pose')
    parser.add_argument('-config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file')
    args = parser.parse_args()

    config_path = args.config if os.path.exists(args.config) else 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    username = config.get('username', '')
    password = config.get('password', '')
    camera_ip = config.get('camera_ip', '')
    port = config.get('port', 554)
    stream_path = config.get('stream_path', '')
    resize_pct = config.get('resize_percentage', 50)

    # Are we doing object detection or pose?
    mode = config.get('mode', 'object')  # "object" or "pose"
    hailo_model_path = config.get('model_path', 'your_model.hailo')

    # Construct RTSP URL
    if stream_path:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}/{stream_path}'
    else:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}'

    # ----------------------------------------------------------
    # Create the pipeline string
    # NOTE: Customize for your environment (decoder, hailonet args, etc.)
    # ----------------------------------------------------------
    # Typically for Hailo, we do something like:
    #   rtspsrc location=... ! queue ! decodebin ! videoconvert ! video/x-raw,format=RGB !
    #       hailonet model_path=... ! hailofilter ! ...
    #
    # For pose estimation, you may have "hailopostprocess" or other elements.
    # The pipeline can differ slightly for object vs. pose. Adjust as needed.

    # Minimal example that goes to an appsink:
    pipeline = (
        f"rtspsrc latency=200 location={rtsp_url} ! "
        "queue ! decodebin ! "
        "videoconvert ! video/x-raw,format=RGB ! "
        f"hailonet name=net model_path={hailo_model_path} ! "
        "hailofilter name=filter ! "
        "appsink emit-signals=True name=sink max-buffers=1 drop=true"
    )

    # ----------------------------------------------------------
    # Create user_data object and pipeline class
    # ----------------------------------------------------------
    user_data = UserAppCallback()
    user_data.use_frame = True  # we want actual frames in callback

    if mode.lower() == "pose":
        # Pose Estimation pipeline
        app = GStreamerPoseEstimationApp(app_callback, user_data, pipeline)
    else:
        # Object Detection pipeline
        app = GStreamerDetectionApp(app_callback, user_data, pipeline)

    # ----------------------------------------------------------
    # Launch pipeline in a separate thread
    # ----------------------------------------------------------
    gst_thread = Thread(target=app.run)
    gst_thread.start()

    # ----------------------------------------------------------
    # Main Loop: display frames in OpenCV
    # ----------------------------------------------------------
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
            app.stop()  # Gracefully stop the pipeline
            break

    gst_thread.join()
    cv2.destroyAllWindows()

#######################################################################
# 5. Entry point
#######################################################################
if __name__ == "__main__":
    main()
