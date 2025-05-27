import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample movie dataset
data = {
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'The Prestige'],
    'genre': ['sci-fi', 'sci-fi', 'sci-fi drama', 'action crime', 'drama mystery'],
    'director': ['Wachowski', 'Nolan', 'Nolan', 'Nolan', 'Nolan']
}

df = pd.DataFrame(data)

# Combine features
df['combined'] = df['genre'] + ' ' + df['director']

# Convert text to vectors
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df['combined'])

# Compute cosine similarity
similarity = cosine_similarity(count_matrix)

# Recommend movies
def recommend(title):
    if title not in df['title'].values:
        return "Movie not found."

    index = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended = [df['title'][i[0]] for i in sorted_scores[1:4]]
    return recommended

# 👇 This part makes it interactive
movie = input("Enter a movie title: ")
print("Recommended movies:", recommend(movie))
