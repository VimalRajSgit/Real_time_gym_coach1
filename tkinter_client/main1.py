import cv2
import mediapipe as mp
import numpy as np
import os
import requests
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
from dotenv import load_dotenv
import threading
import time
import pyttsx3

load_dotenv()

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Flask server URL
FLASK_SERVER_URL = "http://localhost:5000"

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Utility functions (unchanged)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

# BodyPartAngle and TypeOfExercise classes (unchanged)
class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_left_arm(self):
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        l_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        return calculate_angle(l_shoulder, l_elbow, l_wrist)

    def angle_of_the_right_arm(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        r_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        return calculate_angle(r_shoulder, r_elbow, r_wrist)

    def angle_of_the_left_leg(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        l_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
        return calculate_angle(l_hip, l_knee, l_ankle)

    def angle_of_the_right_leg(self):
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        r_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
        return calculate_angle(r_hip, r_knee, r_ankle)

    def angle_of_the_abdomen(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        avg_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        return calculate_angle(l_hip, avg_shoulder, [avg_shoulder[0], avg_shoulder[1] - 0.1])

class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def push_up(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        if status and avg_arm_angle < 70:
            counter += 1
            status = False
        elif not status and avg_arm_angle > 160:
            status = True
        return [counter, status]

    def pull_up(self, counter, status):
        nose = detection_body_part(self.landmarks, "NOSE")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2
        if status and nose[1] > avg_shoulder_y:
            counter += 1
            status = False
        elif not status and nose[1] < avg_shoulder_y:
            status = True
        return [counter, status]

    def squat(self, counter, status):
        left_leg_angle = self.angle_of_the_left_leg()
        right_leg_angle = self.angle_of_the_right_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2
        if status and avg_leg_angle < 70:
            counter += 1
            status = False
        elif not status and avg_leg_angle > 160:
            status = True
        return [counter, status]

    def walk(self, counter, status):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        if status and left_knee[0] > right_knee[0]:
            counter += 1
            status = False
        elif not status and left_knee[0] < right_knee[0]:
            counter += 1
            status = True
        return [counter, status]

    def sit_up(self, counter, status):
        angle = self.angle_of_the_abdomen()
        if status and angle < 55:
            counter += 1
            status = False
        elif not status and angle > 105:
            status = True
        return [counter, status]

    def calculate_exercise(self, exercise_type, counter, status):
        if exercise_type == "push-up":
            return self.push_up(counter, status)
        elif exercise_type == "pull-up":
            return self.pull_up(counter, status)
        elif exercise_type == "squat":
            return self.squat(counter, status)
        elif exercise_type == "walk":
            return self.walk(counter, status)
        elif exercise_type == "sit-up":
            return self.sit_up(counter, status)
        return [counter, status]

class RepCounter:
    def __init__(self):
        self.counter = 0
        self.status = True

    def reset(self):
        self.counter = 0
        self.status = True

    def count_reps(self, frame, exercise_type):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            exercise = TypeOfExercise(results.pose_landmarks.landmark)
            self.counter, self.status = exercise.calculate_exercise(exercise_type.lower(), self.counter, self.status)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return self.counter

class FitnessTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fitness Trainer - Live Workout")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        self.title_font = font.Font(family="Arial", size=16, weight="bold")
        self.normal_font = font.Font(family="Arial", size=12)

        self.workout_plan = {}
        self.current_exercise = None
        self.target_reps = 0
        self.completed_reps = {}
        self.chat_messages = []
        self.rep_counter = RepCounter()
        self.last_coach_update = 0
        self.frame_skip = 0

        self.cap = None
        self.is_running = False
        self.thread = None

        self.create_widgets()
        self.add_chat_message("Coach", "Fetching workout plan...")
        self.fetch_workout_plan()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_label = ttk.Label(main_frame, text="Live Workout", font=self.title_font, background="#1e1e1e", foreground="white")
        video_label.pack(pady=10)

        self.video_frame = ttk.Frame(main_frame, borderwidth=2, relief=tk.SUNKEN)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Coach Chat:", font=self.normal_font, background="#1e1e1e", foreground="white").pack(anchor=tk.W, pady=(10, 5))
        self.chat_display = tk.Text(main_frame, height=10, width=50, font=self.normal_font, state=tk.DISABLED, wrap=tk.WORD, bg="#2e2e2e", fg="white")
        self.chat_display.pack(fill=tk.BOTH, pady=5)

        self.toggle_button = ttk.Button(main_frame, text="Start Workout", command=self.toggle_camera)
        self.toggle_button.pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar(value="Status: Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, background="#1e1e1e", foreground="white")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def fetch_workout_plan(self):
        try:
            response = requests.get(f"{FLASK_SERVER_URL}/get_plan")
            if response.status_code == 200:
                self.workout_plan = response.json()
                if self.workout_plan:
                    self.current_exercise = next(iter(self.workout_plan))
                    self.target_reps = self.workout_plan[self.current_exercise]
                    self.completed_reps[self.current_exercise] = 0
                    self.add_chat_message("Coach", f"Starting {self.target_reps} {self.current_exercise}s!")
                else:
                    self.add_chat_message("Coach", "No workout plan set. Please configure one in the web interface.")
            else:
                self.add_chat_message("Coach", "Error fetching workout plan.")
        except Exception as e:
            self.add_chat_message("Coach", f"Error: {str(e)}")

    def add_chat_message(self, role, message):
        self.chat_messages.append((role, message))
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{role}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        if role == "Coach":
            threading.Thread(target=lambda: (engine.say(message), engine.runAndWait()), daemon=True).start()

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.toggle_button.config(text="Start Workout")
            self.status_var.set("Status: Paused")
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
        else:
            if not self.cap:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                self.status_var.set("Status: Camera error")
                return
            self.is_running = True
            self.toggle_button.config(text="Stop Workout")
            self.status_var.set("Status: In progress")
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.update_frame)
                self.thread.daemon = True
                self.thread.start()

    def update_frame(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Status: Frame error")
                break

            self.frame_skip = (self.frame_skip + 1) % 3
            if self.current_exercise and self.frame_skip == 0:
                previous_count = self.completed_reps.get(self.current_exercise, 0)
                current_count = self.rep_counter.count_reps(frame, self.current_exercise)

                current_time = time.time()
                if current_time - self.last_coach_update >= 2:
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        exercise = TypeOfExercise(results.pose_landmarks.landmark)
                        feedback = self.get_real_time_coaching(exercise, current_count)
                        if feedback:
                            self.root.after(0, lambda: self.add_chat_message("Coach", feedback))
                    self.last_coach_update = current_time

                if current_count > previous_count:
                    self.completed_reps[self.current_exercise] = current_count
                    remaining = max(0, self.target_reps - current_count)
                    if remaining == 0:
                        self.add_chat_message("Coach", f"Done! {self.target_reps} {self.current_exercise}s!")
                        self.workout_plan.pop(self.current_exercise, None)
                        self.completed_reps.pop(self.current_exercise, None)
                        if self.workout_plan:
                            self.current_exercise = next(iter(self.workout_plan))
                            self.target_reps = self.workout_plan[self.current_exercise]
                            self.rep_counter.reset()
                            self.add_chat_message("Coach", f"Next: {self.target_reps} {self.current_exercise}s!")
                        else:
                            self.current_exercise = None
                            self.add_chat_message("Coach", "Workout complete!")
                    else:
                        self.add_chat_message("Coach", f"Great! {remaining} left!")

            if self.current_exercise:
                count = self.completed_reps.get(self.current_exercise, 0)
                cv2.putText(frame, f"Reps: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Exercise: {self.current_exercise}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                img = img.resize((canvas_width, canvas_height), Image.Resampling.NEAREST)

            photo = ImageTk.PhotoImage(image=img)
            self.root.after(0, lambda p=photo: self.update_canvas(p))
            time.sleep(0.01)

    def get_real_time_coaching(self, exercise, current_count):
        if not self.current_exercise or self.target_reps <= 0:
            return None

        remaining = max(0, self.target_reps - current_count)
        prompt = f"Doing {self.current_exercise}s. Reps: {current_count}/{self.target_reps}. "

        if self.current_exercise == "push-up":
            left_arm = exercise.angle_of_the_left_arm()
            if left_arm < 70:
                return "Keep your elbows bent."
            elif left_arm > 160:
                return "Push down now!"
        elif self.current_exercise == "squat":
            left_leg = exercise.angle_of_the_left_leg()
            if left_leg < 70:
                return "Keep your core engaged."
            elif left_leg > 160:
                return "Squat down deeper!"
        elif self.current_exercise == "pull-up":
            return "Make sure your posture is proper."
        return f"Keep going! {remaining} left!"

    def update_canvas(self, photo):
        self.photo = photo
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def on_close(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FitnessTrainerApp(root)
    root.mainloop()