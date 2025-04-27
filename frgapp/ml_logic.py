import pandas as pd # type: ignore
import os
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Load the dataset
def load_dataset():
    csv_path = os.path.join('frgapp', 'recipes.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.dropna(subset=['title', 'ingredients', 'instructions'], inplace=True)
        return df
    else:
        print("CSV not found!")
        return None

# Train the model (TF-IDF on ingredients)
def train_model(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['ingredients'])
    return vectorizer, tfidf_matrix

# Predict the best matching recipes
def get_recommendations(user_ingredients, df, vectorizer, tfidf_matrix):
    # Vectorize the user input ingredients
    user_input_tfidf = vectorizer.transform([user_ingredients])
    
    # Calculate cosine similarity between the user input and the recipes in the dataset
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
    
    # Get the indices of the top 5 most similar recipes
    similar_indices = cosine_sim.argsort()[0, -5:][::-1]
    
    # Get the titles of the most similar recipes
    recommended_recipes = df.iloc[similar_indices]
    return recommended_recipes[['title', 'ingredients']]  # You can adjust what to return based on your data


# Run logic for testing
if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        vectorizer, tfidf_matrix = train_model(df)
        
        # Sample input
        user_input = "chicken, garlic, onion, tomato"
        recommendations = get_recommendations(user_input, df, vectorizer, tfidf_matrix)

        print("\n🔥 Top Recipe Suggestions:")
        for idx, row in recommendations.iterrows():
            print(f"\n🍽️ {row['title']}")
            print(f"📋 Ingredients: {row['ingredients']}")
            print(f"📝 Instructions: {row['instructions'][:100]}...")  # limit for display
# ml_logic.py


# ml_logic.py

# ml_logic.py

def suggest_recipes(user_ingredients):
    import pandas as pd # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity # type: ignore

    # Load the dataset
    df = pd.read_csv('frgapp/recipes.csv')

    # Combine ingredients and instructions
    df['combined'] = df['ingredients'] + ' ' + df['instructions']

    # Vectorize all combined text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    # Vectorize the user input
    user_input_vector = vectorizer.transform([user_ingredients])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(user_input_vector, tfidf_matrix).flatten()

    # Get top N (e.g. 5) matches with non-zero similarity
    top_indices = cosine_sim.argsort()[::-1]  # Descending order
    top_matches = []

    for idx in top_indices:
        if cosine_sim[idx] > 0:
            match = {
                'title': df.iloc[idx]['title'],
                'ingredients': df.iloc[idx]['ingredients'],
                'instructions': df.iloc[idx]['instructions'],
                'score': round(cosine_sim[idx], 2)
            }
            top_matches.append(match)
        if len(top_matches) == 5:
            break

    return top_matches
