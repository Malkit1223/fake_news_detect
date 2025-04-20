from bs4 import BeautifulSoup
import requests
import csv
import time
import threading
import pickle
from pymongo import MongoClient
import bcrypt
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from bson.objectid import ObjectId
from pipeline import train_model  # Import model training function

# ✅ Scraping Functions
def scrape_the_onion():
    """Scrape headlines from The Onion (satirical news)."""
    url = "https://www.theonion.com/"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    headlines = soup.find_all('h2', class_='wp-block-post-title')

    unique_headlines = {headline.get_text(strip=True) for headline in headlines}
    return list(unique_headlines)

def scrape_cnn_news():
    """Scrape CNN World News headlines."""
    url = "https://edition.cnn.com/world"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    headlines = soup.find_all('span', class_='container__headline-text')

    unique_headlines = {headline.get_text(strip=True) for headline in headlines}
    return list(unique_headlines)

def scrape_ndtv_news():
    """Scrape Indian news headlines from NDTV."""
    url = "https://www.ndtv.com/india"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')

    # ✅ Try multiple possible headline classes
    headline_classes = ['NwsLstPg_ttl', 'newsHdng', 'news-headline']
    headlines = []

    for cls in headline_classes:
        elements = soup.find_all('h2', class_=cls)
        for elem in elements:
            text = elem.get_text(strip=True)
            if text:
                headlines.append(text)

    # ✅ Debugging: Print extracted headlines
    print("Extracted NDTV Headlines:", headlines)

    return list(set(headlines))  # Remove duplicates

def save_to_csv(file_name, real_headlines, fake_headlines):
    """Save equal number of real and fake headlines to CSV, alternating labels."""
    existing_headlines = set()

    # Read existing headlines (if file exists)
    try:
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) > 1:
                    existing_headlines.add(row[1])  # Add only headline text
    except FileNotFoundError:
        print(f"CSV file {file_name} not found, creating new file")

    # Filter out duplicates
    real_headlines = [h for h in real_headlines if h not in existing_headlines]
    fake_headlines = [h for h in fake_headlines if h not in existing_headlines]

    # Use the minimum count to ensure equal numbers
    min_count = min(len(real_headlines), len(fake_headlines))
    real_headlines = real_headlines[:min_count]
    fake_headlines = fake_headlines[:min_count]

    # Append new headlines, alternating REAL and FAKE
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header if file was empty
        if not existing_headlines:
            writer.writerow(["label", "text"])
        
        # Write alternating REAL and FAKE headlines
        for i in range(min_count):
            writer.writerow(["REAL", real_headlines[i]])
            writer.writerow(["FAKE", fake_headlines[i]])
            existing_headlines.add(real_headlines[i])
            existing_headlines.add(fake_headlines[i])

    print(f"✅ Added {min_count} REAL and {min_count} FAKE headlines to {file_name} (alternated)")

# ✅ Flask App Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

client = MongoClient('mongodb://localhost:27017/')
db = client['user']
users_collection = db['signup']
mb = client['contact']
message_collection = mb['contactinfo']

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = user_data['name']
        self.email = user_data['email']

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(user_data) if user_data else None

# ✅ Load Model
vector = pickle.load(open("Vetorizer_model.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

# ✅ Routes
@app.route('/')
def home():
    return render_template('index1.html', user=current_user if current_user.is_authenticated else None)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    if request.method == "POST":
        news = request.form['news']
        prediction = model.predict(vector.transform([news]))[0]
        return render_template('prediction.html', prediction_text=f"News headline is -> {prediction}")
    return render_template('prediction.html', user=current_user)

def run_scraping_and_training():
    """Continuously scrape data, save to CSV, and retrain the model."""
    file_name = "news.csv"
    while True:
        print("Starting scraping and model update...")

        real_headlines = scrape_cnn_news() + scrape_ndtv_news()  # CNN & NDTV
        fake_headlines = scrape_the_onion()  # Fake news from The Onion

        # ✅ Debugging: Print scraped data
        print(f"Total Real Headlines: {len(real_headlines)}")
        print(f"Total Fake Headlines: {len(fake_headlines)}")

        if real_headlines and fake_headlines:
            save_to_csv(file_name, real_headlines, fake_headlines)
            train_model(file_name)
            print("Model retrained successfully!")

        time.sleep(200)

# ✅ Start Scraping in Background
thread = threading.Thread(target=run_scraping_and_training)
thread.daemon = True
thread.start()

@app.route('/about')
def about_us():
    return render_template('aboutus.html', user=current_user if current_user.is_authenticated else None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        user_data = users_collection.find_one({"email": email})
        if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data['password']):
            user = User(user_data)
            login_user(user)
            flash("Successfully logged in!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Render contact page and store messages in the database."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_content = request.form['message']

        message_data = {
            "name": name,
            "email": email,
            "message": message_content,
            "timestamp": time.time()
        }

        message_collection.insert_one(message_data)
        flash("Message sent successfully!", "success")
        return redirect(url_for('contact'))

    return render_template('contactus.html', user=current_user if current_user.is_authenticated else None)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('signup'))

        if users_collection.find_one({"email": email}):
            flash("An account with this email already exists. Please log in.", "danger")
            return redirect(url_for('signup'))

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        new_user = {"name": name, "email": email, "password": hashed_password}
        users_collection.insert_one(new_user)
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html', user=current_user)

if __name__ == '__main__':
    app.run(debug=True)