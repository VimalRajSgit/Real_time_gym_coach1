import cv2
import mediapipe as mp
import numpy as np
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

# Initialize pyttsx3 engine with a lock
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
speech_lock = threading.Lock()

FLASK_SERVER_URL = "http://localhost:5000"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

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
        # Stricter: Must go below 60° (down) and above 170° (up)
        if status and avg_arm_angle < 60:
            status = False
        elif not status and avg_arm_angle > 170:
            counter += 1
            status = True
        return [counter, status]

    def pull_up(self, counter, status):
        nose = detection_body_part(self.landmarks, "NOSE")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2
        
        if status and nose[1] > avg_shoulder_y + 0.05:
            status = False
        elif not status and nose[1] < avg_shoulder_y - 0.05:
            counter += 1
            status = True
        return [counter, status]

    def squat(self, counter, status):
        left_leg_angle = self.angle_of_the_left_leg()
        right_leg_angle = self.angle_of_the_right_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2
        
        if status and avg_leg_angle < 60:
            status = False
        elif not status and avg_leg_angle > 170:
            counter += 1
            status = True
        return [counter, status]

    def walk(self, counter, status):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        
        if status and left_knee[0] > right_knee[0] + 0.2:
            status = False
        elif not status and left_knee[0] < right_knee[0] - 0.2:
            counter += 1
            status = True
        return [counter, status]

    def sit_up(self, counter, status):
        angle = self.angle_of_the_abdomen()
        
        if status and angle < 45:
            status = False
        elif not status and angle > 115:
            counter += 1
            status = True
        return [counter, status]

    def bicep_curl(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        
        if status and avg_arm_angle < 50:
            status = False
        elif not status and avg_arm_angle > 150:
            counter += 1
            status = True
        return [counter, status]

    def tricep_curl(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        
        if status and avg_arm_angle > 170:
            status = False
        elif not status and avg_arm_angle < 80:
            counter += 1
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
        elif exercise_type == "bicep-curl":
            return self.bicep_curl(counter, status)
        elif exercise_type == "tricep-curl":
            return self.tricep_curl(counter, status)
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
        self.frame_queue = Queue(maxsize=1)
        self.last_feedback = ""

        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None

        self.create_widgets()
        self.add_chat_message("Coach", "Fetching workout plan...")
        self.fetch_workout_plan()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(10, self.update_gui)

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
                    self.add_chat_message("Coach", f"Let's start with {self.current_exercise}s!")
                else:
                    self.add_chat_message("Coach", "No workout plan set yet. Waiting for exercises...")
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
            threading.Thread(target=self.speak_message, args=(message,), daemon=True).start()

    def speak_message(self, message):
        with speech_lock:
            engine.say(message)
            engine.runAndWait()

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.toggle_button.config(text="Start Workout")
            self.status_var.set("Status: Paused")
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=1.0)
        else:
            if not self.cap:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            if not self.cap.isOpened():
                self.status_var.set("Status: Camera error")
                return
            self.is_running = True
            self.toggle_button.config(text="Stop Workout")
            self.status_var.set("Status: In progress")
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.capture_thread.start()
            self.process_thread.start()

    def capture_frames(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Status: Frame error")
                break
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)
            time.sleep(0.01)

    def process_frames(self):
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if self.current_exercise:
                    previous_count = self.completed_reps.get(self.current_exercise, 0)
                    current_count = self.rep_counter.count_reps(frame, self.current_exercise)

                    current_time = time.time()
                    if current_time - self.last_coach_update >= 2:
                        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if results.pose_landmarks:
                            exercise = TypeOfExercise(results.pose_landmarks.landmark)
                            feedback = self.get_real_time_coaching(exercise)
                            if feedback and feedback != self.last_feedback:
                                self.root.after(0, lambda: self.add_chat_message("Coach", feedback))
                                self.last_feedback = feedback
                        self.last_coach_update = current_time

                    if current_count > previous_count:
                        self.completed_reps[self.current_exercise] = current_count
                        remaining = self.target_reps - current_count
                        if current_count >= self.target_reps:
                            self.add_chat_message("Coach", f"Great job! You've completed {self.current_exercise}s!")
                            self.workout_plan.pop(self.current_exercise, None)
                            self.completed_reps.pop(self.current_exercise, None)
                            if self.workout_plan:
                                self.current_exercise = next(iter(self.workout_plan))
                                self.target_reps = self.workout_plan(self.current_exercise)
                                self.rep_counter.reset()
                                self.add_chat_message("Coach", f"Next up: {self.current_exercise}s!")
                            else:
                                self.current_exercise = None
                                self.add_chat_message("Coach", "Workout complete! Well done!")
                        else:
                            if remaining == 5:
                                self.add_chat_message("Coach", f"Still 5 more")
                            elif remaining == 1:
                                self.add_chat_message("Coach", "Last rep!")

                    if self.current_exercise:
                        count = self.completed_reps.get(self.current_exercise, 0)
                        cv2.putText(frame, f"Reps: {count}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                        cv2.putText(frame, f"Exercise: {self.current_exercise}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    img = img.resize((canvas_width, canvas_height), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(image=img)
                self.root.after(0, lambda p=photo: self.update_canvas(p))

    def update_gui(self):
        if self.is_running:
            self.root.after(10, self.update_gui)

    def get_real_time_coaching(self, exercise):
        if not self.current_exercise:
            return None

        if self.current_exercise == "push-up":
            left_arm = exercise.angle_of_the_left_arm()
            right_arm = exercise.angle_of_the_right_arm()
            avg_arm_angle = (left_arm + right_arm) // 2
            abdomen_angle = exercise.angle_of_the_abdomen()
            if abs(left_arm - right_arm) > 20:
                return "Keep both arms even!"
            elif avg_arm_angle > 170:
                return "Lower yourself more!"
            elif avg_arm_angle < 60:
                return "Push up strong!"
            elif 60 <= avg_arm_angle <= 100:
                return "Go deeper!"
            elif abdomen_angle > 160:
                return "Tighten that core!"
            elif abdomen_angle < 120:
                return "Don’t arch your back!"

        elif self.current_exercise == "squat":
            left_leg = exercise.angle_of_the_left_leg()
            right_leg = exercise.angle_of_the_right_leg()
            avg_leg_angle = (left_leg + right_leg) // 2
            abdomen_angle = exercise.angle_of_the_abdomen()
            if abs(left_leg - right_leg) > 20:
                return "Even out those legs!"
            elif avg_leg_angle > 170:
                return "Bend your knees more!"
            elif 120 <= avg_leg_angle <= 170:
                return "Squat lower!"
            elif avg_leg_angle < 60:
                return "Stand tall!"
            elif 60 <= avg_leg_angle <= 100:
                return "Push those hips back!"
            elif abdomen_angle < 140:
                return "Keep your back straight!"
            elif abdomen_angle > 170:
                return "Engage your core!"

        elif self.current_exercise == "pull-up":
            nose = detection_body_part(exercise.landmarks, "NOSE")
            left_elbow = detection_body_part(exercise.landmarks, "LEFT_ELBOW")
            right_elbow = detection_body_part(exercise.landmarks, "RIGHT_ELBOW")
            avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2
            left_arm = exercise.angle_of_the_left_arm()
            right_arm = exercise.angle_of_the_right_arm()
            if nose[1] > avg_shoulder_y + 0.05:
                return "Chin up higher!"
            elif nose[1] < avg_shoulder_y - 0.05:
                return "Drop smooth!"
            elif abs(left_elbow[0] - right_elbow[0]) > 0.2:
                return "Align elbows!"
            elif left_arm < 60 or right_arm < 60:
                return "Pull harder!"
            elif left_arm > 140 or right_arm > 140:
                return "Bend those arms more!"
            else:
                return "Solid form!"

        elif self.current_exercise == "sit-up":
            abdomen_angle = exercise.angle_of_the_abdomen()
            left_hip = detection_body_part(exercise.landmarks, "LEFT_HIP")
            right_hip = detection_body_part(exercise.landmarks, "RIGHT_HIP")
            if abs(left_hip[1] - right_hip[1]) > 0.1:
                return "Hips stay level!"
            elif abdomen_angle > 115:
                return "Crunch harder!"
            elif 80 <= abdomen_angle <= 115:
                return "Lift higher!"
            elif abdomen_angle < 45:
                return "Nice one!"
            elif 45 <= abdomen_angle <= 80:
                return "Push through!"
            else:
                return "Keep it steady!"

        elif self.current_exercise == "walk":
            right_knee = detection_body_part(exercise.landmarks, "RIGHT_KNEE")
            left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
            left_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
            right_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
            abdomen_angle = exercise.angle_of_the_abdomen()
            if abs(right_knee[0] - left_knee[0]) < 0.2:
                return "Step wider!"
            elif abs(right_ankle[1] - left_ankle[1]) < 0.05:
                return "Lift those feet!"
            elif right_knee[0] > left_knee[0] + 0.2:
                return "Swing that left leg!"
            elif left_knee[0] > right_knee[0] + 0.2:
                return "Move that right leg!"
            elif abdomen_angle < 150:
                return "Stand tall!"
            else:
                return "Nice stride!"

        elif self.current_exercise == "bicep-curl":
            left_arm = exercise.angle_of_the_left_arm()
            right_arm = exercise.angle_of_the_right_arm()
            avg_arm_angle = (left_arm + right_arm) // 2
            if avg_arm_angle > 150:
                return "Curl those arms up!"
            elif 90 <= avg_arm_angle <= 150:
                return "Lift higher!"
            elif avg_arm_angle < 50:
                return "Lower smooth!"
            elif 50 <= avg_arm_angle <= 90:
                return "Full curl!"

        elif self.current_exercise == "tricep-curl":
            left_arm = exercise.angle_of_the_left_arm()
            right_arm = exercise.angle_of_the_right_arm()
            avg_arm_angle = (left_arm + right_arm) // 2
            if avg_arm_angle < 80:
                return "Extend those arms!"
            elif 80 <= avg_arm_angle <= 130:
                return "Push further!"
            elif avg_arm_angle > 170:
                return "Bend back down!"
            elif 130 <= avg_arm_angle <= 170:
                return "Full extension!"

        return None

    def update_canvas(self, photo):
        self.photo = photo
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def on_close(self):
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FitnessTrainerApp(root)
    root.mainloop()
