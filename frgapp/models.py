# frgapp/ml_logic.py

import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Load dataset once at module level
df = pd.read_csv('frgapp/recipes.csv')  # Make sure this file exists

# Fill any missing ingredient data
df['ingredients'] = df['ingredients'].fillna('')

# Create TF-IDF Matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

def suggest_recipes(user_ingredients, top_n=5):
    user_vec = vectorizer.transform([user_ingredients])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    suggestions = df.iloc[top_indices]
    
    results = []
    for _, row in suggestions.iterrows():
        results.append({
            'title': row['title'],  # assuming there's a 'title' column
            'ingredients': row['ingredients']
        })
    return results
