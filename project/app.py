from flask import Flask, render_template, request, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import numpy as np
import re
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hackathon.db'
db = SQLAlchemy(app)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with desired SBERT model


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    profile_description = db.Column(db.Text, nullable=True)

# Hackathon data (mocked for demonstration)
hackathons = [
    {"name": "AI Challenge 2024", "description": "An AI-focused hackathon for creative solutions."},
    {"name": "Web Innovators Hack", "description": "Create next-gen web apps in this competitive event."},
    {"name": "Blockchain Revolution", "description": "Solve blockchain challenges and create DApps."}
]

# Helper Functions

def generate_color():
    """
    Generate a random color in hexadecimal format.
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def highlight_keywords(text, keywords, class_name):
    """
    Highlight keywords in the given text by wrapping them with a span tag
    with a CSS class for styling.
    """
    for keyword in keywords:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)  # Match whole words only
        text = pattern.sub(
            lambda match: '<span class="{0}">{1}</span>'.format(class_name, match.group(0)),
            text
        )
    return text

def extract_keywords_from_profile(profile_description, max_features=10):
    """
    Extract meaningful keywords from the user's profile description using TF-IDF.
    Args:
        profile_description (str): User's inputted profile description.
        max_features (int): Maximum number of keywords to extract.
    Returns:
        set: Set of extracted keywords.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform([profile_description])
    return set(vectorizer.get_feature_names_out())


def normalize_text(text):
    """
    Normalize text by converting to lowercase and removing special characters.
    Args:
        text (str): Input text to normalize.
    Returns:
        str: Normalized text.
    """
    return re.sub(r'[^\w\s]', '', text.lower())

def recommend_hackathons(profile_description):
    # Normalize and extract keywords from the user's profile description
    normalized_profile = normalize_text(profile_description)
    profile_keywords = extract_keywords_from_profile(normalized_profile)

    # Get profile embedding using SBERT with sliding window technique
    profile_embedding = get_text_embedding(normalized_profile, sbert_model)

    recommendations = []

    for hack in hackathons:
        # Normalize hackathon description
        normalized_hack_description = normalize_text(hack["description"])
        hack_embedding = get_text_embedding(normalized_hack_description, sbert_model)

        # Compute cosine similarity
        similarity_score = np.dot(profile_embedding, hack_embedding) / (
            np.linalg.norm(profile_embedding) * np.linalg.norm(hack_embedding)
        )

        # Highlight overlapping keywords in the hackathon description
        overlapping_keywords = profile_keywords & set(normalized_hack_description.split())
        highlighted_hack_description = highlight_keywords(
            hack["description"], overlapping_keywords, "highlight-hackathon"
        )

        recommendations.append({
            "name": hack["name"],
            "original_description": hack["description"],
            "highlighted_description": highlighted_hack_description,
            "score": round(similarity_score * 100, 2),
            "keywords": list(overlapping_keywords),  # Relevant keywords for explanation
        })

    # Highlight keywords in the user's profile description
    highlighted_profile = highlight_keywords(profile_description, profile_keywords, "highlight-profile")

    # Sort recommendations by similarity score (descending)
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations, highlighted_profile


def process_long_text(text, max_length=128, overlap=32):
    """
    Split text into chunks using a sliding window approach.
    Args:
        text (str): Input long text.
        max_length (int): Maximum length of each chunk.
        overlap (int): Overlap between chunks for contextual continuity.
    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_length
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_length - overlap  # Sliding window ensures overlap between chunks
    return chunks


def get_text_embedding(text, model):
    """
    Generate embeddings for a text using SBERT with hierarchical and sliding window techniques.
    Args:
        text (str): Input text (could be long).
        model: Preloaded SBERT model.
    Returns:
        np.array: Averaged embedding for the entire text.
    """
    # Process text into chunks using sliding window
    chunks = process_long_text(text)
    # Generate embeddings for each chunk
    chunk_embeddings = np.array([model.encode(chunk) for chunk in chunks])
    # Average the embeddings to get a single embedding for the text
    return np.mean(chunk_embeddings, axis=0)


@app.route('/explanation')
def explanation():
    if 'user_id' not in session:
        return redirect('/')
    user = User.query.get(session['user_id'])
    if not user.profile_description:
        return redirect('/profile')
    
    chunks = process_long_text(user.profile_description)
    embeddings = [sbert_model.encode(chunk) for chunk in chunks]
    averaged_embedding = np.mean(embeddings, axis=0)

    chunk_lengths = [len(chunk.split()) for chunk in chunks]

    visualization_data = {
        "chunks": chunks,
        "chunk_lengths": chunk_lengths,
        "embedding_shape": list(averaged_embedding.shape),  # Convert tuple to list
        "num_chunks": len(chunks),
    }

    return render_template(
        'explanation.html',
        user=user,
        visualization_data=visualization_data
    )


# Routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect('/dashboard')
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            return "Username already exists!"
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect('/')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect('/dashboard')
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect('/')
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        user.profile_description = request.form['profile_description']
        db.session.commit()
        return redirect('/dashboard')
    return render_template('profile.html', user=user)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/')
    user = User.query.get(session['user_id'])
    if not user.profile_description:
        return redirect('/profile')
    
    # Use hierarchical and sliding window techniques in `recommend_hackathons`
    recommendations, highlighted_profile = recommend_hackathons(user.profile_description)
    
    return render_template(
        'dashboard.html',
        user=user,
        recommendations=recommendations,
        highlighted_profile=highlighted_profile
    )

# Initialize Database
with app.app_context():
    db.create_all()

# Run App
if __name__ == '__main__':
    app.run(debug=True)
