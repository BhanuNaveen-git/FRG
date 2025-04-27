from django.shortcuts import render
from .ml_logic import load_dataset, train_model, get_recommendations, suggest_recipes

# Load and train once at the start
df = load_dataset()
vectorizer, tfidf_matrix = train_model(df)

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'POST':
        user_ingredients = request.POST.get('ingredients')
        
        if not user_ingredients:
            return render(request, 'result.html', {'error': "Please enter ingredients!"})

        # Call suggest_recipes and assume it returns a list of dictionaries directly
        recipes = suggest_recipes(user_ingredients)

        return render(request, 'result.html', {
            'input': user_ingredients,
            'recipes': recipes,
            'ingredients': user_ingredients,
            'results': [recipe['title'] for recipe in recipes]  # Optional for your cards
        })

    return render(request, 'result.html')


    return render(request, 'result.html')  # For GET request
