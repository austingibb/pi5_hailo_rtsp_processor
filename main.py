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
# For object detection:
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# For standard pose estimation (we’ll subclass it):
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# Helper pipeline-building functions from Hailo (used in GStreamerPoseEstimationApp)
# Typically found in gstreamer_helper_pipelines.py or similar:
from hailo_apps_infra.gstreamer_helper_pipelines import (
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE
)

############################################################################
# 1. COCO Keypoints Map
############################################################################
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

############################################################################
# 2. User-defined callback class
############################################################################
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

############################################################################
# 3. Our GStreamer callback function
############################################################################
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment frame count
    user_data.increment()
    frame_number = user_data.get_count()

    # Get caps
    fmt, width, height = get_caps_from_pad(pad)
    if not user_data.use_frame or (fmt is None):
        return Gst.PadProbeReturn.OK

    # Convert to numpy
    frame = get_numpy_from_buffer(buffer, fmt, width, height)

    # Pull Hailo detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    for detection in detections:
        detection_count += 1
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        # Draw bounding box
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}",
                    (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # If pose landmarks exist
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) > 0:
            pts = landmarks[0].get_points()
            for name, idx in KEYPOINTS_DICT.items():
                point = pts[idx]
                px = int((point.x() * bbox.width()) + bbox.xmin())
                py = int((point.y() * bbox.height()) + bbox.ymin())
                cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

    # Put debug text
    cv2.putText(frame,
                f"Frame #{frame_number} / Detections: {detection_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Convert to BGR for display
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    user_data.set_frame(frame_bgr)

    return Gst.PadProbeReturn.OK


############################################################################
# 4. Subclass GStreamerPoseEstimationApp for a custom RTSP pipeline
############################################################################
from hailo_apps_infra.gstreamer_app import GStreamerApp
class MyCustomRTSPPoseEstimationApp(GStreamerPoseEstimationApp):
    """
    A subclass that overrides get_pipeline_string() to use rtspsrc instead
    of the default SOURCE_PIPELINE logic.
    """

    def get_pipeline_string(self):
        # 1. Construct rtspsrc portion (with decodebin -> videoconvert -> RGB)
        # You can tweak 'latency', 'queue' usage, etc. as needed:
        rtspsrc_part = (
            f'rtspsrc location="{self.video_source}" latency=200 ! '
            'queue ! decodebin ! videoconvert ! video/x-raw,format=RGB ! '
        )

        # 2. Set up the Hailo inference pipeline.
        #    Typically, we have to define:
        #      - self.hef_path
        #      - self.post_process_so (if needed)
        #      - self.post_process_function
        #      - self.batch_size
        #    The parent class may have already set some defaults, or you can set them in __init__.
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size
        )
        # Turn it into a single pipeline string element
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)  # or whichever class ID you want to track
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        # 3. Concatenate everything
        pipeline_string = (
            f"{rtspsrc_part}"         #  => "rtspsrc ... ! queue ! decodebin ! videoconvert ! video/x-raw,format=RGB ! "
            f"{infer_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        print("[Custom RTSP Pose Pipeline] = ", pipeline_string)
        return pipeline_string

    # If you want to do custom initialization, override __init__ as well
    # but be sure to call the parent's constructor with just 2 arguments:
    #
    # def __init__(self, app_callback, user_data):
    #     super().__init__(app_callback, user_data)
    #     # set self.batch_size, self.hef_path, etc., if not done in parent's init
    #
    # The parent constructor uses command-line args, so you might set
    # them also. Or just rely on "self.hef_path" being set from the parent's logic.


############################################################################
# 5. Main function
############################################################################
def main():
    # ----------------------------------------------------------
    # Parse CLI arguments
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(description='Hailo + OpenCV: Object or Pose over RTSP')
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

    mode = config.get('mode', 'object')  # "object" or "pose"
    hailo_model_path = config.get('model_path', 'your_model.hailo')

    # Build RTSP URL
    if stream_path:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}/{stream_path}'
    else:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}'

    # For object detection, we can still do the old approach with a pipeline:
    pipeline = (
        f"rtspsrc latency=200 location={rtsp_url} ! "
        "queue ! decodebin ! "
        "videoconvert ! video/x-raw,format=RGB ! "
        f"hailonet name=net model_path={hailo_model_path} ! "
        "hailofilter name=filter ! "
        "appsink emit-signals=True name=sink max-buffers=1 drop=true"
    )

    # ----------------------------------------------------------
    # Create user_data (callback storage) and choose the app class
    # ----------------------------------------------------------
    user_data = UserAppCallback()
    user_data.use_frame = True

    if mode.lower() == "pose":
        # Use our custom RTSP Pose Estimation class
        # NOTE: This expects you to set `self.video_source`, `self.hef_path`, etc.
        # Possibly you can pass them in via command-line. We'll do it manually:
        app = MyCustomRTSPPoseEstimationApp(app_callback, user_data)

        # We can override some parent attributes:
        app.video_source = rtsp_url              # Tells our custom pipeline to use rtspsrc with that URL
        app.hef_path = hailo_model_path          # So inference uses the correct model
        app.batch_size = 1                       # Example
        # If you have a post-process .so or a function name for YOLO pose:
        # app.post_process_so = "/path/to/your_pose_postprocess.so"
        # app.post_process_function = "filter_letterbox"
        # etc.

    else:
        # Object detection approach with a direct pipeline
        app = GStreamerDetectionApp(app_callback, user_data, pipeline)

    # ----------------------------------------------------------
    # Run in a separate thread
    # ----------------------------------------------------------
    gst_thread = Thread(target=app.run)
    gst_thread.start()

    # ----------------------------------------------------------
    # Main display loop
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
            app.stop()
            break

    gst_thread.join()
    cv2.destroyAllWindows()

############################################################################
# 6. Entry point
############################################################################
if __name__ == "__main__":
    main()
