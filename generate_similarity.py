import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading movies_list.pkl...")
movies = pickle.load(open('movies_list.pkl', 'rb'))
print(f"Loaded: {movies.shape}")

print("Vectorizing tags with CountVectorizer...")
cv = CountVectorizer(max_features=10000, stop_words='english')
vectors = cv.fit_transform(movies['tags'].values.astype('U'))
print(f"Vectors shape: {vectors.shape}")

print("Computing cosine similarity (this may take 1-2 mins)...")
similarity = cosine_similarity(vectors)
print(f"Similarity shape: {similarity.shape}")

print("Saving similarity.pkl...")
pickle.dump(similarity, open('similarity.pkl', 'wb'))
print("Done! similarity.pkl created.")
