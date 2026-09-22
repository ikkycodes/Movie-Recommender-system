import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve the data files relative to this script so the tool works no matter
# which directory it is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(BASE_DIR, "movies_list.pkl")
SIM_PATH = os.path.join(BASE_DIR, "similarity.pkl")

print(f"Loading {MOVIES_PATH}...")
with open(MOVIES_PATH, "rb") as f:
    movies = pickle.load(f)
print(f"Loaded: {movies.shape}")

print("Vectorizing tags with CountVectorizer...")
cv = CountVectorizer(max_features=10000, stop_words='english')
vectors = cv.fit_transform(movies['tags'].values.astype('U'))
print(f"Vectors shape: {vectors.shape}")

print("Computing cosine similarity (this may take 1-2 mins)...")
similarity = cosine_similarity(vectors)
print(f"Similarity shape: {similarity.shape}")

print(f"Saving {SIM_PATH}...")
with open(SIM_PATH, "wb") as f:
    pickle.dump(similarity, f)
print("Done! similarity.pkl created.")
