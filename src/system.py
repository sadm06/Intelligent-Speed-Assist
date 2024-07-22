import os
import sys
import time
from datetime import timedelta

import cv2
import pandas as pd
import pygame
import torch

from display import DISPLAY_SIZE, display_dash_active, display_dash_error, display_dash_inactive, display_dash_init, \
    display_dash_standby, display_dash_warning_auditive, display_dash_warning_haptic, display_dash_warning_visual
from processing import inference, initialize_models, show_fps, show_sign

DETECTION_MODEL_PATH = 'models/detection/train1/weights/best.pt'
CLASSIFICATION_MODEL_PATH = 'models/classification/train2/best.pth'
CLASSES_PATH = 'datasets/GTSRB/classes.csv'

FIVE_HERTZ = 0.2  # Frequency of the processing in Hz
INFERENCE_SIZE = 640  # Size of the frame for inference
END_OF_LIMITS = 32  # Class ID for the end of speed limits sign
DISTANCE = 100  # Minimum distance to the speed camera in meters
DECELERATION = 7  # Average deceleration of a car on a normal surface with standard braking behavior
OPTICAL_DELAY = -1  # Delay in seconds until optical warning is shown (Set to -1 to use speed dependent delay)
ACOUSTIC_DELAY = 2  # Delay in seconds after optical warning
HAPTIC_DELAY = 4  # Delay in seconds after optical warning
CAP_FPS = True  # Set to True to use the input video's fps, False to use an uncapped fps
STATE_CHANGE_THRESHOLD = 30  # Speed threshold for state change
MAX_SPEED = 180  # Maximum speed for active state


def load_class_names(classes_path):
    meta_data = pd.read_csv(classes_path)
    class_names = {}
    for _, row in meta_data.iterrows():
        class_names[row['ClassId']] = row['ClassName']
    return class_names


def load_sign_images(classes_path):
    meta_data = pd.read_csv(classes_path)
    sign_images = {}
    for _, row in meta_data.iterrows():
        image_path = os.path.join('datasets/GTSRB', row['Path'])
        class_id = row['ClassId']
        image = cv2.imread(image_path)
        sign_images[class_id] = image
    return sign_images


def load_speed_data():
    df = pd.read_csv(metadata_path, header=None)
    df['time'] = pd.to_datetime(df.iloc[:, 0], format='%Y:%m:%d %H:%M:%SZ') + timedelta(
        hours=2)  # Add 2 hours to convert to local time
    df['speed'] = df.iloc[:, 1]
    video_start_time = df['time'].iloc[0]
    return df, video_start_time


def load_speed_limits(classes_path):
    meta_data = pd.read_csv(classes_path)
    speed_limits = {i: -1 for i in range(43)}
    for _, row in meta_data.iterrows():
        if pd.notnull(row['SpeedLimit']):
            speed_limits[row['ClassId']] = int(row['SpeedLimit'])
    return speed_limits


def get_video_properties():
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_scale = original_width / INFERENCE_SIZE
    y_scale = original_height / INFERENCE_SIZE
    fps = cap.get(cv2.CAP_PROP_FPS)
    return original_width, original_height, x_scale, y_scale, fps


def get_current_speed(current_time, df):
    matched_row = df[df['time'] == current_time]
    if not matched_row.empty:
        return int(matched_row['speed'].values[0])
    else:
        return -1


def update_warnings(current_speed, speed_limit, num_frames_speeding, fps):
    warning = 'none'
    if current_speed <= 100:
        speed_limit += 3
    else:
        speed_limit += speed_limit * 0.03
    if current_speed > speed_limit:
        num_frames_speeding += 1
        num_sec_speeding = num_frames_speeding / fps
        if OPTICAL_DELAY == -1:
            current_speed_mps = current_speed * 1000 / 3600
            speed_limit_mps = speed_limit * 1000 / 3600
            time_to_break = (current_speed_mps - speed_limit_mps) / DECELERATION
            time_to_blitzer = DISTANCE / current_speed_mps
            time_until_warning = time_to_blitzer - time_to_break
        else:
            time_until_warning = OPTICAL_DELAY

        if num_sec_speeding >= time_until_warning:
            warning = 'optical'
        if num_sec_speeding >= time_until_warning + ACOUSTIC_DELAY:
            warning = 'acoustic'
        if num_sec_speeding >= time_until_warning + HAPTIC_DELAY:
            warning = 'haptic'

    elif current_speed <= speed_limit:
        num_frames_speeding = 0
    return num_frames_speeding, warning


class SystemState:
    INIT = 'init'
    STANDBY = 'standby'
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    ERROR = 'error'
    DEBUG = 'debug'


class State:
    def __init__(self):
        self.context = context

    def handle(self):
        pass


class InitState(State):
    def handle(self):
        print("Initializing system...")
        if torch.cuda.is_available():
            self.context.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.context.device = torch.device("mps")
        else:
            self.context.device = torch.device("cpu")

        self.context.class_names = load_class_names(CLASSES_PATH)
        self.context.sign_images = load_sign_images(CLASSES_PATH)
        self.context.speed_limits = load_speed_limits(CLASSES_PATH)
        if self.context.assist_mode:
            self.context.speed_df, self.context.video_start_time = load_speed_data()

        print("Loading models...")
        self.context.detection_model, self.context.classification_model = (
            initialize_models(self.context.device, DETECTION_MODEL_PATH, CLASSIFICATION_MODEL_PATH,
                              len(self.context.class_names)))
        print("Models loaded successfully")

        (self.context.original_width, self.context.original_height, self.context.x_scale, self.context.y_scale,
         self.context.fps) = get_video_properties()
        self.context.target_frame_time = 1.0 / self.context.fps
        self.context.frequency = int(self.context.fps * FIVE_HERTZ)

        if not self.context.cap.isOpened():
            print("Error opening video file.")
            self.context.state = ErrorState()
            return

        self.context.prev_frame_time = time.time()
        self.context.frame_count, self.context.num_frames_speeding = 0, 0
        self.context.last_class = END_OF_LIMITS
        self.context.speed_limit = self.context.speed_limits[END_OF_LIMITS]
        self.context.class_id, self.context.class_conf = -1, 0.0
        self.context.time_until_warning = 0
        print("System initialized successfully.")
        if self.context.assist_mode:
            self.context.state = StandbyState()
        else:
            self.context.state = DebugState()


class StandbyState(State):
    def handle(self):
        current_time = (self.context.current_time - timedelta(hours=2)).strftime("%H:%M")
        current_date = self.context.current_time.strftime("%d.%m.%Y")
        display_dash_standby(self.context.screen, self.context.current_speed, current_date, current_time)
        frame = self.context.frame

        cv2.imshow('Dashboard Camera', frame)


class ActiveState(State):
    def handle(self):
        current_time = (self.context.current_time - timedelta(hours=2)).strftime("%H:%M")
        current_date = self.context.current_time.strftime("%d.%m.%Y")
        frame = self.context.frame
        inference_frame = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))

        if self.context.frame_count % self.context.frequency == 0:
            frame, self.context.class_id, self.context.class_conf = (
                inference(frame, inference_frame, self.context.x_scale, self.context.y_scale, self.context.class_names,
                          self.context.detection_model, self.context.classification_model, self.context.device, False))
            if self.context.class_conf > 0.95 and (self.context.class_id <= 8 or self.context.class_id == 32):
                self.context.last_class = self.context.class_id
                self.context.speed_limit = self.context.speed_limits[self.context.last_class]

        self.context.num_frames_speeding, warning = update_warnings(self.context.current_speed,
                                                                    self.context.speed_limit,
                                                                    self.context.num_frames_speeding, self.context.fps)
        match warning:
            case 'none':
                display_dash_active(self.context.screen, self.context.current_speed, self.context.speed_limit,
                                    current_date, current_time)
            case 'optical':
                display_dash_warning_visual(self.context.screen, self.context.current_speed, self.context.speed_limit,
                                            current_date, current_time)
            case 'acoustic':
                display_dash_warning_auditive(self.context.screen, self.context.current_speed, self.context.speed_limit,
                                              current_date, current_time)
            case 'haptic':
                display_dash_warning_haptic(self.context.screen, self.context.current_speed, self.context.speed_limit,
                                            current_date, current_time)

        cv2.imshow('Dashboard Camera', frame)


class InactiveState(State):
    def handle(self):
        current_time = (self.context.current_time - timedelta(hours=2)).strftime("%H:%M")
        current_date = self.context.current_time.strftime("%d.%m.%Y")
        display_dash_inactive(self.context.screen, self.context.current_speed, current_date, current_time)
        cv2.imshow('Dashboard Camera', self.context.inactive_image)
        self.context.prev_frame_time = time.time()


class ErrorState(State):
    def handle(self):
        if assist_mode:
            display_dash_error(self.context.screen)
        cv2.imshow('Dashboard Camera', self.context.inactive_image)


class DebugState(State):
    def handle(self):
        frame = self.context.frame

        inference_frame = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))

        frame, self.context.class_id, _ = inference(frame, inference_frame, self.context.x_scale, self.context.y_scale,
                                                    self.context.class_names, self.context.detection_model,
                                                    self.context.classification_model, self.context.device, True)

        frame, self.context.prev_frame_time = show_fps(frame, self.context.prev_frame_time)
        if 0 <= self.context.class_id < len(self.context.sign_images):
            frame = show_sign(frame, self.context.sign_images[self.context.class_id], self.context.original_width,
                              self.context.original_height)

        cv2.imshow('Dashboard Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)


class System:
    def __init__(self):
        self.state = None
        self.frequency = 0
        self.speed_limits = None
        self.original_width = 0
        self.original_height = 0
        self.sign_images = None
        self.device = None
        self.class_names = None
        self.classification_model = None
        self.detection_model = None
        self.y_scale = 0
        self.x_scale = 0
        self.current_speed = 0
        self.speed_df = None
        self.video_start_time = None
        self.fps = 0
        self.target_frame_time = None
        self.frame = None
        self.current_time = None
        self.under_threshold = 0
        self.assist_mode = assist_mode
        self.inactive_image = cv2.imread('assets/no_camera_feed.jpg')
        if assist_mode:
            cv2.imshow('Dashboard Camera', self.inactive_image)
            pygame.init()
            pygame.display.set_caption("Car Display")
            self.screen_info = pygame.display.Info()
            os.environ['SDL_VIDEO_WINDOW_POS'] = (f"{self.screen_info.current_w - DISPLAY_SIZE[0]},"
                                                  f"{self.screen_info.current_h - DISPLAY_SIZE[1]}")
            self.screen = pygame.display.set_mode(DISPLAY_SIZE)
            display_dash_init(self.screen)
            pygame.display.flip()
            self.metadata_path = metadata_path
        self.cap = cap
        self.video_path = video_path
        self.frame_count = 0

    def run(self):
        self.state = InitState()
        while True:
            start_time = time.time()
            success, self.frame = cap.read()
            if not success:
                self.state = ErrorState()
            else:
                self.frame_count += 1

            self.state.handle()

            if self.assist_mode:
                pygame.display.flip()
                elapsed_seconds = int(self.frame_count / self.fps)
                self.current_time = self.video_start_time + timedelta(seconds=elapsed_seconds)
                speed_at_time = get_current_speed(self.current_time, self.speed_df)
                if speed_at_time >= 0:
                    self.current_speed = speed_at_time

                if STATE_CHANGE_THRESHOLD <= self.current_speed <= MAX_SPEED:
                    self.under_threshold = 0
                    self.state = ActiveState() if not isinstance(self.state, InactiveState) else self.state
                elif self.current_speed <= STATE_CHANGE_THRESHOLD or self.current_speed > MAX_SPEED:
                    self.under_threshold += 1
                    if self.under_threshold / self.fps >= 3:
                        self.state = StandbyState() if not isinstance(self.state, InactiveState) else self.state

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            self.state = InactiveState() if not isinstance(self.state,
                                                                           InactiveState) else StandbyState()
                        elif event.key == pygame.K_q:
                            return

            if CAP_FPS:
                processing_time = time.time() - start_time
                if processing_time < self.target_frame_time:
                    time.sleep(self.target_frame_time - processing_time)


if __name__ == "__main__":
    assist_mode = True
    video_path = ''
    metadata_path = ''
    if len(sys.argv) < 2:
        print("Usage: python system.py <video_path> <metadata_path>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        video_path = sys.argv[1]
        metadata_path = video_path.rsplit('.', 1)[0] + '.csv'
    elif len(sys.argv) > 2:
        video_path = sys.argv[1]
        metadata_path = sys.argv[2]
    if not os.path.exists(video_path):
        print("Video file does not exist.")
        sys.exit(1)
    if not os.path.exists(metadata_path):
        print("No metadata file found. Entering debug mode.")
        assist_mode = False
    cap = cv2.VideoCapture(video_path)
    context = System()
    context.run()
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
