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
from queue import Queue

load_dotenv()

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Faster speech rate
engine.setProperty('volume', 0.9)

# Groq API configuration
def get_llama_response(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: API key not configured"
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a lively Gym coach. Give short, unique, real-sounding motivational shouts based on rep count and target. "
                    "Examples: Rep 1: 'Hit it hard!', Rep 2-3: 'Stay tough!', Mid-reps: 'Grind it out!', Near last: 'You’re so close!', Last: 'Smash it!' "
                    "More options: 'Power up!', 'Dig in!', 'Keep rocking!', 'One more, champ!', 'End strong!' "
                    "Avoid repetition, keep it fresh and punchy, max 5-7 words."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,  # Higher for more variety
        "max_tokens": 20     # Slightly increased for natural phrasing
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=0.3)  # Even faster timeout
        return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else "Error"
    except Exception:
        return "Keep going!"  # Fallback for speed

# Flask server URL
FLASK_SERVER_URL = "http://localhost:5000"

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Utility functions
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
        return {
            "push-up": self.push_up,
            "pull-up": self.pull_up,
            "squat": self.squat,
            "walk": self.walk,
            "sit-up": self.sit_up
        }.get(exercise_type, lambda c, s: [c, s])(counter, status)

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
        self.chat_queue = Queue()
        self.rep_counter = RepCounter()
        self.last_rep_count = -1  # Track last announced rep
        self.last_response = ""   # Track last response to avoid repeats

        self.cap = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

        self.create_widgets()
        self.add_chat_message("Coach", "Getting your plan...")
        self.fetch_workout_plan()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_chat_worker()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text="Live Workout", font=self.title_font, background="#1e1e1e", foreground="white").pack(pady=5)

        self.video_frame = ttk.Frame(main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(self.video_frame, bg="black", width=320, height=240)
        self.canvas.pack(fill=tk.BOTH, expand=False)
        self.canvas.bind("<Configure>", self.resize_canvas)

        ttk.Label(main_frame, text="Coach Chat:", font=self.normal_font, background="#1e1e1e", foreground="white").pack(anchor=tk.W, pady=(5, 2))
        self.chat_display = tk.Text(main_frame, height=6, width=50, font=self.normal_font, state=tk.DISABLED, wrap=tk.WORD, bg="#2e2e2e", fg="white")
        self.chat_display.pack(fill=tk.X, pady=2)

        self.toggle_button = ttk.Button(main_frame, text="Start Workout", command=self.toggle_camera)
        self.toggle_button.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, background="#1e1e1e", foreground="white").pack(side=tk.BOTTOM, fill=tk.X)

    def resize_canvas(self, event):
        if self.is_running:
            self.canvas.config(width=event.width, height=event.height)

    def toggle_camera(self):
        with self.lock:
            if self.is_running:
                self.is_running = False
                self.toggle_button.config(text="Start Workout")
                self.status_var.set("Status: Paused")
                self.video_frame.pack(fill=tk.BOTH, expand=False)
                self.canvas.pack(fill=tk.BOTH, expand=False)
                self.canvas.config(width=320, height=240)
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
                self.video_frame.pack(fill=tk.BOTH, expand=True)
                self.canvas.pack(fill=tk.BOTH, expand=True)
                if self.thread is None or not self.thread.is_alive():
                    self.thread = threading.Thread(target=self.update_frame, daemon=True)
                    self.thread.start()

    def update_frame(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Status: Frame error")
                break

            if self.current_exercise:
                current_count = self.rep_counter.count_reps(frame, self.current_exercise)
                with self.lock:
                    if current_count != self.completed_reps.get(self.current_exercise, 0):
                        self.completed_reps[self.current_exercise] = current_count
                        remaining = self.target_reps - current_count
                        if current_count > self.last_rep_count:  # Only announce new reps
                            prompt = self.get_prompt(current_count, remaining)
                            threading.Thread(target=self.async_llama_response, args=(prompt,), daemon=True).start()
                            self.last_rep_count = current_count
                        if remaining <= 0:
                            self.workout_plan.pop(self.current_exercise, None)
                            self.completed_reps.pop(self.current_exercise, None)
                            self.last_rep_count = -1
                            if self.workout_plan:
                                self.current_exercise = next(iter(self.workout_plan))
                                self.target_reps = self.workout_plan[self.current_exercise]
                                self.rep_counter.reset()
                                threading.Thread(target=self.async_llama_response, args=(f"Next: {self.current_exercise}!"), daemon=True).start()
                            else:
                                self.current_exercise = None
                                threading.Thread(target=self.async_llama_response, args=("Workout done, awesome!"), daemon=True).start()

                count = self.completed_reps.get(self.current_exercise, 0)
                cv2.putText(frame, f"Reps: {count}/{self.target_reps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Exercise: {self.current_exercise}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                if (canvas_width / canvas_height) > aspect_ratio:
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                offset_x = (canvas_width - new_width) // 2
                offset_y = (canvas_height - new_height) // 2
            else:
                offset_x, offset_y = 0, 0

            photo = ImageTk.PhotoImage(image=img)
            self.root.after(0, self.update_canvas, photo, offset_x, offset_y)
            time.sleep(0.005)  # Further reduced delay for real-time

    def fetch_workout_plan(self):
        try:
            response = requests.get(f"{FLASK_SERVER_URL}/get_plan", timeout=1)  # Faster timeout
            if response.status_code == 200:
                self.workout_plan = response.json()
                if self.workout_plan:
                    self.current_exercise = next(iter(self.workout_plan))
                    self.target_reps = self.workout_plan[self.current_exercise]
                    self.completed_reps[self.current_exercise] = 0
                    threading.Thread(target=self.async_llama_response, args=(f"Let’s crush {self.current_exercise}!"), daemon=True).start()
                else:
                    self.add_chat_message("Coach", "No plan ready.")
            else:
                self.add_chat_message("Coach", "Plan fetch failed.")
        except Exception as e:
            self.add_chat_message("Coach", f"Error: {str(e)}")

    def add_chat_message(self, role, message):
        with self.lock:
            self.chat_queue.put((role, message))

    def start_chat_worker(self):
        def worker():
            while True:
                role, message = self.chat_queue.get()
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.insert(tk.END, f"{role}: {message}\n")
                self.chat_display.see(tk.END)
                self.chat_display.config(state=tk.DISABLED)
                if role == "Coach" and message != self.last_response:  # Avoid repeating voice
                    engine.say(message)
                    engine.runAndWait()
                    self.last_response = message
                self.chat_queue.task_done()
        threading.Thread(target=worker, daemon=True).start()

    def async_llama_response(self, prompt):
        response = get_llama_response(prompt)
        self.add_chat_message("Coach", response)

    def get_prompt(self, current_count, remaining):
        if not self.current_exercise or self.target_reps <= 0:
            return None
        rep_stage = (
            "rep 1" if current_count == 1 else
            "reps 2-3" if current_count in [2, 3] else
            "mid-reps" if current_count > 3 and remaining > 2 else
            "near last" if remaining <= 2 and remaining > 0 else
            "last rep" if remaining == 0 else ""
        )
        return f"{self.current_exercise}: {current_count}/{self.target_reps} - {rep_stage}"

    def update_canvas(self, photo, offset_x, offset_y):
        with self.lock:
            self.photo = photo
            self.canvas.delete("all")
            self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo)

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