from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Store workout plans in memory (replace with a database in production)
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
        return jsonify({"status": "success", "message": f"Added {reps} {exercise}s"})
    return jsonify({"status": "error", "message": "Invalid exercise or reps"}), 400

@app.route('/get_plan', methods=['GET'])
def get_plan():
    return jsonify(workout_plans)

@app.route('/reset_plan', methods=['POST'])
def reset_plan():
    workout_plans.clear()
    return jsonify({"status": "success", "message": "Workout plan reset"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)