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

# Hailo helpers
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

########################################################################
# 1. User-defined callback class
########################################################################
class UserAppCallback(app_callback_class):
    """
    Inherit from hailo_apps_infra.hailo_rpi_common.app_callback_class
    so we can store frames & custom variables.
    """
    def __init__(self):
        super().__init__()
        self.lock = Lock()
        self.latest_frame = None  # We'll store the most recent overlaid frame here.

    def set_frame(self, frame):
        """Store the latest overlaid frame (thread-safe)."""
        with self.lock:
            self.latest_frame = frame

    def get_frame(self):
        """Retrieve the latest overlaid frame (thread-safe)."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


########################################################################
# 2. Callback function that GStreamer calls for every buffer (frame)
########################################################################
def app_callback(pad, info, user_data):
    """
    pad:  The GStreamer pad giving us data
    info: The probe info
    user_data: Our UserAppCallback instance
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Count frames (for example)
    user_data.increment()
    frame_number = user_data.get_count()

    # Extract caps (format, width, height)
    fmt, width, height = get_caps_from_pad(pad)
    if (not user_data.use_frame) or (fmt is None):
        return Gst.PadProbeReturn.OK

    # Convert buffer to numpy
    frame = get_numpy_from_buffer(buffer, fmt, width, height)
    # frame from Hailo is typically in RGB, we can confirm that in your pipeline’s "video/x-raw,format=RGB" config.

    # Retrieve detections from Hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Overlay detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()  # x_min, y_min, x_max, y_max
        confidence = detection.get_confidence()

        # For instance, we only draw if it's a "person"
        if label == "person":
            detection_count += 1

            # If you have tracking info
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = track[0].get_id() if len(track) == 1 else 0

            # Draw bounding box on the frame
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text_label = f"ID:{track_id} {label} {confidence:.2f}"
            cv2.putText(frame, text_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # You can overlay any debug info you want
    cv2.putText(frame, f"Frame #{frame_number}, Detections: {detection_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert back to BGR if needed for display in standard OpenCV window
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Store the frame in user_data so the main thread can display it
    user_data.set_frame(frame_bgr)

    return Gst.PadProbeReturn.OK


########################################################################
# 3. Main function
########################################################################
def main():
    # -----------------------------------------------------------
    # Parse CLI arguments (same approach as your RTSP script)
    # -----------------------------------------------------------
    parser = argparse.ArgumentParser(description='RTSP Stream Capture + Hailo Pose/Detection Overlay')
    parser.add_argument('-config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file')
    args = parser.parse_args()

    config_path = args.config if os.path.exists(args.config) else 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    username = config['username']
    password = config['password']
    camera_ip = config['camera_ip']
    port = config['port']
    stream_path = config.get('stream_path', '')
    resize_percentage = config.get('resize_percentage', 50)

    # Construct the RTSP URL
    if stream_path:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}/{stream_path}'
    else:
        rtsp_url = f'rtsp://{username}:{password}@{camera_ip}:{port}'

    # -----------------------------------------------------------
    # Create our user_data callback class
    # -----------------------------------------------------------
    user_data = UserAppCallback()
    # We want actual frames in our callback
    user_data.use_frame = True

    # -----------------------------------------------------------
    # Prepare the Hailo GStreamer detection pipeline
    # -----------------------------------------------------------
    #
    # Typically, you have something like:
    #
    #  rtspsrc location=rtsp://... ! queue ! decodebin ! videoconvert ! video/x-raw,format=RGB ! \
    #       hailonet model_path=<your_model.hailo> ! hailofilter ! appsink ...
    #
    # The exact pipeline depends on your model, Hailo plugin versions, etc.
    # Adjust as needed (latency, decodebin, etc.). Also note if you need
    #  "hailopostprocess" or other plugin elements in your chain.
    #
    # The key is that at the end we have an appsink with "emit-signals=True"
    # so GStreamerDetectionApp can attach the callback.
    #
    pipeline = (
        f"rtspsrc latency=200 location={rtsp_url} ! "
        "queue ! decodebin ! "
        "videoconvert ! video/x-raw,format=RGB ! "
        # Replace with your actual Hailo elements:
        "hailonet name=net model_path=your_model.hailo ! "
        "hailofilter name=filter ! "
        "appsink emit-signals=True name=sink max-buffers=1 drop=true"
    )

    # -----------------------------------------------------------
    # Initialize GStreamerDetectionApp
    #   - The first arg is the callback function
    #   - The second arg is the user_data object
    #   - The third arg is the pipeline string
    # -----------------------------------------------------------
    app = GStreamerDetectionApp(app_callback, user_data, pipeline)

    # We'll run the GStreamer pipeline in a separate thread
    gst_thread = Thread(target=app.run)
    gst_thread.start()

    # -----------------------------------------------------------
    # MAIN LOOP: Retrieve frames from user_data, show with OpenCV
    # -----------------------------------------------------------
    cv2.namedWindow("RTSP + Hailo Detections", cv2.WINDOW_NORMAL)
    while True:
        frame_bgr = user_data.get_frame()
        if frame_bgr is not None:
            # Optionally resize for display:
            height, width = frame_bgr.shape[:2]
            new_width = int(width * (resize_percentage / 100))
            new_height = int(height * (resize_percentage / 100))
            resized_frame = cv2.resize(frame_bgr, (new_width, new_height))

            cv2.imshow("RTSP + Hailo Detections", resized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Graceful shutdown
            app.stop()  # Tells GStreamerDetectionApp to stop main pipeline loop
            break

    gst_thread.join()
    cv2.destroyAllWindows()

########################################################################
# 4. Entry point
########################################################################
if __name__ == "__main__":
    main()
