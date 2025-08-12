from flask import Flask, render_template, request, redirect, session, flash, jsonify 
from markupsafe import Markup
import sqlite3
import requests
import random
import json
import os   
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Sample formulas dictionary
aptitude_formulas = {
    "time-and-work": """
        <ul>
            <li>If A can do a piece of work in n days, then A's 1 day work = 1/n</li>
            <li>If A is n times as good a worker as B, then ratio of work = A:B = n:1</li>
            <li>Total work = Work rate × Time</li>
        </ul>
    """,
    "profit-and-loss": """
        <ul>
            <li>Profit = SP - CP</li>
            <li>Loss = CP - SP</li>
            <li>Profit % = (Profit / CP) × 100</li>
            <li>Loss % = (Loss / CP) × 100</li>
        </ul>
    """,
    "percentages": """
        <ul>
            <li>Percentage = (Value / Total Value) × 100</li>
            <li>Increase% = ((New - Original)/Original) × 100</li>
        </ul>
    """,
}

# List of LeetCode topics supported by the API
leetcode_topics = [
    "array", "string", "hash-table", "math", "dynamic-programming",
    "sorting", "greedy", "depth-first-search", "breadth-first-search",
    "tree", "binary-search", "linked-list", "stack", "queue",
    "heap", "graph", "two-pointers", "divide-and-conquer",
    "sliding-window", "design", "topological-sort", "matrix",
    "trie", "quickselect", "backtracking", "bit-manipulation"
]

learning_materials = {
    "Percentages": Markup("""
        <ul>
            <li><b>Percentage Formula:</b> (Part/Whole) × 100</li>
            <li><b>Increase by x%:</b> Final = Initial × (1 + x/100)</li>
            <li><b>Decrease by x%:</b> Final = Initial × (1 - x/100)</li>
        </ul>
    """),
    "Time and Work": Markup("""
        <ul>
            <li><b>Work Formula:</b> Work = Rate × Time</li>
            <li><b>If A can do a job in x days, A's 1-day work = 1/x</li>
            <li><b>Combined work:</b> If A and B work together, 1/x + 1/y = 1/total</li>
        </ul>
    """),
    "Profit and Loss": Markup("""
        <ul>
            <li><b>Profit:</b> Selling Price - Cost Price</li>
            <li><b>Profit %:</b> (Profit / Cost Price) × 100</li>
            <li><b>Loss %:</b> (Loss / Cost Price) × 100</li>
        </ul>
    """)
}

# Sample aptitude questions
aptitude_questions = {
    "Percentages": [
        {
            "question": "What is 25% of 200?",
            "options": ["25", "50", "75", "100"],
            "answer": "50"
        },
        {
            "question": "A value increases from 80 to 100. What is the percentage increase?",
            "options": ["20%", "25%", "30%", "40%"],
            "answer": "25%"
        },
        {
            "question": "What is 40% of 150?",
            "options": ["50", "60", "70", "80"],
            "answer": "60"
        },
        {
            "question": "If 30% of a number is 90, what is the number?",
            "options": ["270", "300", "280", "250"],
            "answer": "300"
        },
        {
            "question": "A man's salary is increased by 20% and then decreased by 20%. What is the net change?",
            "options": ["4% decrease", "4% increase", "No change", "2% decrease"],
            "answer": "4% decrease"
        },
        {
            "question": "A number is first increased by 10% and then increased again by 20%. What is the overall percentage increase?",
            "options": ["30%", "32%", "28%", "25%"],
            "answer": "32%"
        },
        {
            "question": "If 60 is 75% of a number, what is the number?",
            "options": ["70", "75", "80", "85"],
            "answer": "80"
        },
        {
            "question": "What percentage of 1 hour is 45 minutes?",
            "options": ["50%", "60%", "75%", "90%"],
            "answer": "75%"
        },
        {
            "question": "If a shirt is marked at ₹1200 and a discount of 25% is offered, what is the selling price?",
            "options": ["₹800", "₹900", "₹1000", "₹950"],
            "answer": "₹900"
        },
        {
            "question": "A population increases from 20,000 to 25,000. What is the percentage increase?",
            "options": ["20%", "25%", "30%", "15%"],
            "answer": "25%"
        }
    ],
    "Time and Work": [
        {
            "question": "If A can do a work in 10 days, how much work does A do in 1 day?",
            "options": ["1/10", "10", "1/5", "None"],
            "answer": "1/10"
        }
    ]
}

# ML Classes
class PerformancePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def prepare_training_data(self, username=None):
        """Prepare training data from database"""
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        
        # Get historical aptitude data
        if username:
            query = """
                SELECT username, topic, score, total_questions, 
                       timestamp, COUNT(*) as attempt_count
                FROM aptitude_progress 
                WHERE username = ?
                GROUP BY username, topic
            """
            cursor.execute(query, (username,))
        else:
            query = """
                SELECT username, topic, score, total_questions, 
                       timestamp, COUNT(*) as attempt_count
                FROM aptitude_progress 
                GROUP BY username, topic
            """
            cursor.execute(query)
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) < 5:  # Need minimum data points
            return None, None
        
        # Prepare features and targets
        features = []
        targets = []
        
        for row in data:
            username, topic, score, total_q, timestamp, attempts = row
            
            # Features: [previous_score, attempts_count, topic_encoded]
            prev_score = (score / total_q) * 100 if total_q > 0 else 0
            topic_encoded = self.encode_topic(topic)
            
            features.append([prev_score, attempts, topic_encoded])
            targets.append(prev_score)
        
        return np.array(features), np.array(targets)
    
    def encode_topic(self, topic):
        """Simple topic encoding"""
        topic_mapping = {
            "Percentages": 1,
            "Time and Work": 2, 
            "Profit and Loss": 3
        }
        return topic_mapping.get(topic, 0)
    
    def train_model(self):
        """Train the model with available data"""
        X, y = self.prepare_training_data()
        
        if X is None:
            return False
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except:
            return False
    
    def predict_performance(self, username, topic):
        """Predict user's performance on a topic"""
        if not self.is_trained:
            if not self.train_model():
                return None
        
        # Get user's historical data
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(score*100.0/total_questions) as avg_score,
                   COUNT(*) as attempts
            FROM aptitude_progress 
            WHERE username = ?
        """, (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0] is None:
            return 65  # Default prediction for new users
        
        # Prepare prediction features
        avg_score, attempts = result
        topic_encoded = self.encode_topic(topic)
        
        features = np.array([[avg_score, attempts, topic_encoded]])
        prediction = self.model.predict(features)[0]
        
        # Clamp prediction between 0-100
        return max(0, min(100, prediction))


class DifficultyRecommender:
    def __init__(self):
        self.model = LogisticRegression()
        self.difficulties = ['Easy', 'Medium', 'Hard']
    
    def recommend_difficulty(self, username, topic):
        """Recommend difficulty based on user performance"""
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        
        # Get coding attempt history
        cursor.execute("""
            SELECT difficulty, completed, time_spent, COUNT(*) as attempts
            FROM coding_attempts 
            WHERE username = ? AND topic = ?
            GROUP BY difficulty
        """, (username, topic))
        
        history = cursor.fetchall()
        conn.close()
        
        if not history:
            return "Easy"  # Start with Easy for new users
        
        # Calculate success rates
        difficulty_stats = {}
        for diff, completed, avg_time, attempts in history:
            success_rate = completed / attempts if attempts > 0 else 0
            difficulty_stats[diff] = {
                'success_rate': success_rate,
                'attempts': attempts,
                'avg_time': avg_time or 1800  # Default 30 min
            }
        
        # Simple rule-based recommendation
        if 'Easy' in difficulty_stats:
            easy_stats = difficulty_stats['Easy']
            if easy_stats['success_rate'] >= 0.8 and easy_stats['avg_time'] < 1200:  # 20 min
                if 'Medium' in difficulty_stats:
                    medium_stats = difficulty_stats['Medium']
                    if medium_stats['success_rate'] >= 0.6:
                        return "Hard"
                    else:
                        return "Medium"
                else:
                    return "Medium"
            else:
                return "Easy"
        
        return "Easy"


class TopicRecommender:
    def __init__(self):
        self.topics = ["Percentages", "Time and Work", "Profit and Loss"]
    
    def suggest_next_topic(self, username):
        """Suggest next topic to study"""
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        
        topic_performance = {}
        
        for topic in self.topics:
            cursor.execute("""
                SELECT AVG(score*100.0/total_questions) as avg_score,
                       COUNT(*) as attempts,
                       MAX(timestamp) as last_attempt
                FROM aptitude_progress 
                WHERE username = ? AND topic = ?
            """, (username, topic))
            
            result = cursor.fetchone()
            
            if result[0] is not None:
                topic_performance[topic] = {
                    'avg_score': result[0],
                    'attempts': result[1],
                    'last_attempt': result[2]
                }
        
        conn.close()
        
        if not topic_performance:
            return "Percentages"  # Start with basics
        
        # Decision tree logic
        weak_topics = [topic for topic, stats in topic_performance.items() 
                      if stats['avg_score'] < 70]
        
        if weak_topics:
            # Recommend weakest topic
            return min(weak_topics, 
                      key=lambda t: topic_performance[t]['avg_score'])
        
        # If all topics are strong, recommend least practiced
        return min(topic_performance.keys(), 
                  key=lambda t: topic_performance[t]['attempts'])


# Initialize the ML models
performance_predictor = PerformancePredictor()
difficulty_recommender = DifficultyRecommender()
topic_recommender = TopicRecommender()

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Create aptitude progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS aptitude_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            topic TEXT NOT NULL,
            score INTEGER,
            total_questions INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create notes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    
    # Create ML-specific tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coding_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            time_spent INTEGER DEFAULT 0,
            completed BOOLEAN DEFAULT 0,
            hints_used INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_learning_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_duration INTEGER,
            questions_attempted INTEGER,
            topics_covered TEXT,
            performance_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Commit all changes and close connection ONCE at the end
    conn.commit()
    conn.close()


# Fetch a random LeetCode question by topic and difficulty
def get_random_leetcode_question(topic_slug, difficulty):
    try:
        with open('leetcode_questions.json', 'r') as f:
            all_questions = json.load(f)

        filtered = [
            q for q in all_questions
            if topic_slug in q.get("tags", [])
               and q.get("difficulty", "").lower() == difficulty.lower()
        ]

        if not filtered:
            return None

        selected = random.choice(filtered)
        return {
            "title": selected["title"],
            "slug": selected["titleSlug"],
            "difficulty": selected["difficulty"],
            "url": f"https://leetcode.com/problems/{selected['titleSlug']}/"
        }

    except Exception as e:
        print(f"Error loading local question: {e}")
        return None


@app.route('/')
def home():
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect('/dashboard')
        else:
            flash("Invalid username or password.")
            return redirect('/login')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect('/register')

        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                           (username, email, password))
            conn.commit()
            flash("Registration successful! Please log in.")
            return redirect('/login')
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
            return redirect('/register')
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    else:
        return redirect('/login')


@app.route('/practice')
def practice():
    if 'username' not in session:
        return redirect('/login')
    return render_template('practice.html', username=session['username'], topics=leetcode_topics)


@app.route('/get-dynamic-question')
def get_dynamic_question():
    topic = request.args.get('topic')
    difficulty = request.args.get('difficulty')

    if not topic or not difficulty:
        return jsonify({'error': 'Missing topic or difficulty'}), 400

    question = get_random_leetcode_question(topic, difficulty)
    if not question:
        return jsonify({'error': 'No question found'}), 404

    return jsonify(question)


@app.route('/aptitude')
def aptitude():
    if 'username' not in session:
        return redirect('/login')

    topics = list(aptitude_questions.keys())
    progress_dict = {topic: None for topic in topics}

    return render_template('aptitude.html',
                           username=session['username'],
                           topics=topics,
                           progress_dict=progress_dict)


@app.route('/aptitude/learn/<topic_slug>')
def learn_topic(topic_slug):
    if 'username' not in session:
        return redirect('/login')

    content = aptitude_formulas.get(topic_slug, "<p>Content coming soon for this topic.</p>")
    topic_title = topic_slug.replace('-', ' ').title()

    return render_template('learn.html', username=session['username'], topic=topic_title, content=Markup(content))


@app.route('/learn/<topic>')
def learn(topic):
    if 'username' not in session:
        return redirect('/login')

    topic_title = topic.replace("-", " ").title()
    content = learning_materials.get(topic_title)

    if content:
        return render_template('learn.html', topic_name=topic_title, content=content)
    else:
        flash("Learning material not available for this topic.")
        return redirect('/aptitude')


# Notes Routes
@app.route('/notes')
def notes():
    if 'username' not in session:
        return redirect('/login')
    
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, content, category, created_at, updated_at 
        FROM notes 
        WHERE username = ? 
        ORDER BY updated_at DESC
    """, (session['username'],))
    
    user_notes = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries for easier template handling
    notes_list = []
    for note in user_notes:
        notes_list.append({
            'id': note[0],
            'title': note[1],
            'content': note[2],
            'category': note[3],
            'created_at': note[4],
            'updated_at': note[5]
        })
    
    return render_template('notes.html', username=session['username'], notes=notes_list)


@app.route('/notes/create', methods=['GET', 'POST'])
def create_note():
    if 'username' not in session:
        return redirect('/login')
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        category = request.form['category']
        
        if not title or not content:
            flash("Title and content are required.")
            return redirect('/notes/create')
        
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notes (username, title, content, category) 
            VALUES (?, ?, ?, ?)
        """, (session['username'], title, content, category))
        conn.commit()
        conn.close()
        
        flash("Note created successfully!")
        return redirect('/notes')
    
    return render_template('create_note.html', username=session['username'])


@app.route('/notes/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    if 'username' not in session:
        return redirect('/login')
    
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        category = request.form['category']
        
        if not title or not content:
            flash("Title and content are required.")
            return redirect(f'/notes/edit/{note_id}')
        
        cursor.execute("""
            UPDATE notes 
            SET title = ?, content = ?, category = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ? AND username = ?
        """, (title, content, category, note_id, session['username']))
        conn.commit()
        conn.close()
        
        flash("Note updated successfully!")
        return redirect('/notes')
    
    # GET request - fetch note data
    cursor.execute("""
        SELECT id, title, content, category 
        FROM notes 
        WHERE id = ? AND username = ?
    """, (note_id, session['username']))
    
    note = cursor.fetchone()
    conn.close()
    
    if not note:
        flash("Note not found.")
        return redirect('/notes')
    
    note_dict = {
        'id': note[0],
        'title': note[1],
        'content': note[2],
        'category': note[3]
    }
    
    return render_template('edit_note.html', username=session['username'], note=note_dict)


@app.route('/notes/delete/<int:note_id>')
def delete_note(note_id):
    if 'username' not in session:
        return redirect('/login')
    
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ? AND username = ?", (note_id, session['username']))
    conn.commit()
    conn.close()
    
    flash("Note deleted successfully!")
    return redirect('/notes')


@app.route('/solve/<topic>', methods=['GET', 'POST'])
def solve(topic):
    if 'username' not in session:
        return redirect('/login')

    topic_title = topic.replace("-", " ").title()
    questions = aptitude_questions.get(topic_title, [])
    total = len(questions)

    if 'quiz_data' not in session or session['quiz_data'].get('topic') != topic_title:
        session['quiz_data'] = {
            'topic': topic_title,
            'answers': [],
            'index': 0,
            'start_time': datetime.now().isoformat()  # Track start time
        }

    quiz = session['quiz_data']
    index = quiz['index']

    if index >= total:
        score = sum([1 for i, q in enumerate(questions)
                     if quiz['answers'][i] == q['answer']])
        percent = int((score / total) * 100)
        
        # Store results in database for ML
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO aptitude_progress (username, topic, score, total_questions)
            VALUES (?, ?, ?, ?)
        """, (session['username'], topic_title, score, total))
        conn.commit()
        conn.close()
        
        session.pop('quiz_data', None)
        return f"<h2>Quiz Finished! You scored {score}/{total} ({percent}%)</h2><br><a href='/ml-insights'>View AI Insights</a><br><a href='/aptitude'>Back to Aptitude</a>"

    if request.method == 'POST':
        selected = request.form.get('answer')
        if selected:
            quiz['answers'].append(selected)
            quiz['index'] += 1
            session['quiz_data'] = quiz
        return redirect(f"/solve/{topic}")

    current_question = questions[index]
    progress = int((index / total) * 100)

    return render_template("solve.html",
                           topic_name=topic_title,
                           question=current_question,
                           current_index=index,
                           total_questions=total,
                           progress=progress)


@app.route('/ml-insights')
def ml_insights():
    if 'username' not in session:
        return redirect('/login')
    
    username = session['username']
    
    # Get AI recommendations
    next_topic = topic_recommender.suggest_next_topic(username)
    
    # Get performance prediction for the recommended topic
    predicted_score = performance_predictor.predict_performance(username, next_topic)
    
    # Get difficulty recommendation for coding
    recommended_difficulty = difficulty_recommender.recommend_difficulty(username, "array")
    
    # Get user stats for dashboard
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as total_attempts,
               AVG(score*100.0/total_questions) as avg_performance
        FROM aptitude_progress 
        WHERE username = ?
    """, (username,))
    
    stats = cursor.fetchone()
    conn.close()
    
    total_attempts = stats[0] if stats[0] else 0
    avg_performance = round(stats[1], 1) if stats[1] else 0
    
    return render_template('ml_insights.html',
                         username=username,
                         next_topic=next_topic,
                         predicted_score=round(predicted_score, 1) if predicted_score else "Not available",
                         recommended_difficulty=recommended_difficulty,
                         total_attempts=total_attempts,
                         avg_performance=avg_performance)


@app.route('/track-coding-attempt', methods=['POST'])
def track_coding_attempt():
    """Track coding practice attempts for ML"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    username = session['username']
    topic = data.get('topic', 'unknown')
    difficulty = data.get('difficulty', 'Easy')
    time_spent = data.get('time_spent', 0)
    completed = data.get('completed', False)
    
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO coding_attempts (username, topic, difficulty, time_spent, completed)
        VALUES (?, ?, ?, ?, ?)
    """, (username, topic, difficulty, time_spent, completed))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
