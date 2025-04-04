from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import subprocess
import threading

load_dotenv()

app = Flask(__name__)


workout_plans = {}

@app.route('/')
def index():
    return render_template('index.html', plans=workout_plans)

@app.route('/add_exercise', methods=['POST'])
def add_exercise():
    data = request.json
    exercise = data.get('exercise')
    reps = int(data.get('reps', 0))
    if exercise and reps > 0:
        workout_plans[exercise] = reps
        # Launch m.py in a new process each time a new exercise is added
        threading.Thread(target=start_fitness_app, daemon=True).start()
        return jsonify({"status": "success", "message": f"Added {reps} {exercise}s"})
    return jsonify({"status": "error", "message": "Invalid exercise or reps"}), 400

@app.route('/get_plan', methods=['GET'])
def get_plan():
    return jsonify(workout_plans)

@app.route('/reset_plan', methods=['POST'])
def reset_plan():
    workout_plans.clear()
    return jsonify({"status": "success", "message": "Workout plan reset"})

def start_fitness_app():
    """Function to start m.py as a separate process"""
    # Path to m.py in the tkinter_client folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # C:\projects\fitness_trainer_app
    m_py_path = os.path.join(root_dir, 'tkinter_client', 'm.py')
    
    # Run m.py as a separate process from its own directory
    subprocess.Popen(['python', m_py_path], cwd=os.path.join(root_dir, 'tkinter_client'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
