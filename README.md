<<<<<<< HEAD
🎬 Movie Recommendation System

A Content-Based Filtering ML Project


---

📌 Overview

This project is a Movie Recommendation System that suggests the top 5 most similar movies based on a user-selected title. It uses content-based filtering with text similarity techniques to generate meaningful and personalized movie recommendations.


---

🚀 Features

Recommends Top 5 similar movies

Uses CountVectorizer for text feature extraction

Cosine Similarity for similarity scoring

Cleaned & processed dataset

Interactive Streamlit interface

Pre-computed movies list & similarity matrix stored using Pickle


---

🧠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

CountVectorizer

Cosine Similarity

Pickle

Streamlit


---

📂 Project Structure

(Updated as per your actual files)

movie-recommender/
│── app.py                 
│── main.py                
│── main.ipynb             
│── dataset.csv            
│── movies_list.pkl        
│── similarity.pkl         
└── .ipynb_checkpoints/    


---

⚙️ How It Works

1. Dataset is cleaned and a combined tags column is created.


2. CountVectorizer converts text into numerical vectors.


3. Cosine Similarity computes similarity scores between movies.


4. Results (movies list + similarity matrix) are saved as .pkl files.


5. Streamlit UI (app.py) loads these files and recommends movies instantly.


---

▶️ Run the App

Run the following command in your terminal:

streamlit run app.py

---

📚 Learnings

This project strengthened skills in:

Data preprocessing & cleaning

Text-based feature engineering

Similarity-based ML concepts

Saving & loading ML assets using Pickle

Building interactive ML applications with Streamlit

---

⭐ Demo

(Add screenshots or GIFs here if you want!)


---

🤝 Contributing

Feel free to open issues or submit pull requests.


---

📜 License

This project is open-source under the MIT License.
=======
# Movie-Recommender-system
A content-based Movie Recommendation System that suggests the top 5 similar movies using CountVectorizer and Cosine Similarity. Built on the Netflix dataset with extensive cleaning, feature engineering, and model saving via Pickle. Includes an interactive Streamlit UI for smooth user experience.
>>>>>>> 70439c6ba98bf5429f90ae09ebde536bb7785154
