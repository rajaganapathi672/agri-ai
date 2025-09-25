from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 role TEXT NOT NULL)''')
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 ('admin', 'admin123', 'admin'))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

init_db()

# PyTorch Model Definition
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load models
try:
    crop_model = joblib.load("models/crop_yield_model.pkl")
except:
    from sklearn.ensemble import RandomForestRegressor
    crop_model = RandomForestRegressor()
    joblib.dump(crop_model, "models/crop_yield_model.pkl")

try:
    disease_model = PlantDiseaseCNN()
    disease_model.load_state_dict(torch.load('models/plant_disease_model.pt'))
    disease_model.eval()
except:
    disease_model = PlantDiseaseCNN()
    torch.save(disease_model.state_dict(), 'models/plant_disease_model.pt')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop-yield', methods=['GET', 'POST'])
def crop_yield():
    soil_types = {
        "Alluvial": 1,
        "Black": 2,
        "Red": 3,
        "Laterite": 4,
        "Mountain": 5,
        "Desert": 6
    }

    fertilizer_types = {
        "Urea": 1,
        "DAP": 2,
        "Potash": 3,
        "Compost": 4,
        "Organic": 5
    }

    crop_types = {
        "Wheat": 1,
        "Rice": 2,
        "Maize": 3,
        "Sugarcane": 4,
        "Cotton": 5,
        "Barley": 6
    }

    prediction = None

    if request.method == 'POST':
        try:
            soil_type_name = request.form['soil_type']
            fertilizer_type_name = request.form['fertilizer_type']
            crop_type_name = request.form['crop_type']

            soil_type = soil_types[soil_type_name]
            fertilizer_type = fertilizer_types[fertilizer_type_name]
            crop_type = crop_types[crop_type_name]

            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])

            input_data = pd.DataFrame([[soil_type, rainfall, temperature, humidity, fertilizer_type, crop_type]],
                columns=['soil_type', 'rainfall', 'temperature', 'humidity', 'fertilizer_type', 'crop_type'])

            prediction = crop_model.predict(input_data)[0]
        except Exception as e:
            flash('Error in prediction: ' + str(e), 'error')

    return render_template('crop_yield.html',
                           prediction=round(prediction, 2) if prediction else None,
                           soil_types=soil_types.keys(),
                           fertilizer_types=fertilizer_types.keys(),
                           crop_types=crop_types.keys())

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                image = Image.open(filepath).convert('RGB')
                image = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = disease_model(image)
                    _, predicted = torch.max(outputs.data, 1)
                
                disease_classes = ['Healthy', 'Disease A', 'Disease B']
                result = disease_classes[predicted.item()]
                
                return render_template('disease_detection.html', 
                                      prediction=result, 
                                      image_url=url_for('static', filename='images/uploads/' + filename))
            except Exception as e:
                flash('Error in disease detection: ' + str(e), 'error')
    
    return render_template('disease_detection.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            
            if user[3] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin/dashboard.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
