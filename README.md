🎬 Movie Recommendation System

A Content-Based Filtering ML Project

---

📌 Overview

This project is a Movie Recommendation System that suggests the top 5 most similar movies based on a user-selected title. It uses content-based filtering with text similarity techniques to generate meaningful and personalized movie recommendations.

Built on the Netflix dataset with extensive cleaning, feature engineering, and model saving via Pickle. Includes an interactive Streamlit UI for a smooth user experience.

---

🚀 Features

- Recommends the Top 5 similar movies
- Uses CountVectorizer for text feature extraction
- Cosine Similarity for similarity scoring
- Cleaned & processed dataset
- Interactive Streamlit interface
- Posters, release year, IMDb rating and plot fetched from the OMDb API
- Runs without the large `similarity.pkl` file — the matrix is rebuilt from the `tags` column on demand

---

🧠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- CountVectorizer
- Cosine Similarity
- Pickle
- Streamlit
- OMDb API

---

📂 Project Structure

```
movie-recommender/
│── app.py                    # Streamlit UI + recommendation logic
│── generate_similarity.py    # Optional: caches the full similarity matrix
│── main.py                   # Quick dataset preview
│── main.ipynb                # Data cleaning & feature engineering notebook
│── dataset.csv               # Working dataset
│── movies_list.pkl           # 10,000 movies (id, title, tags)
│── similarity.pkl            # Optional ~800 MB matrix (not stored in this repo)
│── requirements.txt          # Dependencies for the app and the devcontainer
└── .devcontainer/            # GitHub Codespaces configuration
```

---

⚙️ How It Works

1. Dataset is cleaned and a combined `tags` column is created.
2. CountVectorizer converts text into numerical vectors.
3. Cosine Similarity computes similarity scores between movies.
4. Results are saved to `movies_list.pkl`, plus `similarity.pkl` when generated.
5. Streamlit UI (`app.py`) loads these files and recommends movies instantly.

### About `similarity.pkl`

`similarity.pkl` is a ~800 MB dense matrix, which is larger than GitHub's 100 MB
per-file limit, so **it is not stored in this repository** and is listed in
`.gitignore`. No download step is needed: when the file is absent, `app.py`
computes cosine similarity from `movies_list.pkl["tags"]` using the same
CountVectorizer settings, so the app works from a fresh clone.

To cache the full matrix to disk instead, run this once (~1–2 minutes):

```
python generate_similarity.py
```

---

▶️ Run the App

```
pip install -r requirements.txt
streamlit run app.py
```

### GitHub Codespaces

This repository includes a `.devcontainer`, so a newly created Codespace installs
the dependencies automatically. If your Codespace was created before
`requirements.txt` existed, run these commands once in its terminal:

```
git pull
pip install -r requirements.txt
```

---

📚 Learnings

This project strengthened skills in:

- Data preprocessing & cleaning
- Text-based feature engineering
- Similarity-based ML concepts
- Saving & loading ML assets using Pickle
- Building interactive ML applications with Streamlit

---

⭐ Demo

(Add screenshots or GIFs here if you want!)

---

🤝 Contributing

Feel free to open issues or submit pull requests.

---

📜 License

This project is open-source under the MIT License.
