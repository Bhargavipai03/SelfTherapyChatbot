import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from chatbot_core import TherapyChatbot, AnonymityManager # Add this import


app = Flask(__name__)
CORS(app)

# ---------- DATABASE CONFIG ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'zenmind.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- DATABASE MODELS ----------
class Booking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    psychologist = db.Column(db.String(100), nullable=False)
    datetime = db.Column(db.String(100), nullable=False)
    reason = db.Column(db.Text, nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# ---------- INITIALIZE COMPONENTS ----------
chatbot = TherapyChatbot()

anonymity_manager = AnonymityManager()

# ---------- ROUTES ----------
@app.route('/')
def home():
    return "✅ ZenMind API is running."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'response': 'Please enter a message.', 'error': False}), 400

        clean_message = anonymity_manager.sanitize_input(user_message)


        # ✅ ALWAYS RETURN RESPONSE STRING, NOT A TUPLE
        response_text = chatbot.generate_response(clean_message)
        return jsonify({'response': response_text, 'error': False})

    except Exception as e:
        print("[SERVER ERROR]", e)
        return jsonify({'response': '⚠️ Something went wrong on the server.', 'error': True, 'details': str(e)}), 500

@app.route('/book', methods=['POST'])
def book_appointment():
    try:
        data = request.get_json()
        required = ['name', 'psychologist', 'datetime', 'reason']
        if not all(field in data for field in required):
            return jsonify({"message": "Missing fields"}), 400

        new_booking = Booking(
            name=data['name'],
            psychologist=data['psychologist'],
            datetime=data['datetime'],
            reason=data['reason']
        )
        db.session.add(new_booking)
        db.session.commit()
        return jsonify({"message": "✅ Appointment booked successfully!"}), 2000
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/appointments', methods=['GET'])
def view_appointments():
    try:
        bookings = Booking.query.all()
        results = [{
            "id": b.id,
            "name": b.name,
            "psychologist": b.psychologist,
            "datetime": b.datetime,
            "reason": b.reason
        } for b in bookings]
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already registered.'}), 400

        hashed_pw = generate_password_hash(data['password'])
        new_user = User(name=data['name'], email=data['email'], password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Signup successful!'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': 'Signup failed.', 'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        user = User.query.filter_by(email=data['email']).first()
        if user and check_password_hash(user.password, data['password']):
            return jsonify({'success': True, 'message': 'Login successful'}), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': 'Login failed', 'error': str(e)}), 500

# ---------- START ----------
if __name__ == '__main__':
    app.run(debug=True, port=5000)