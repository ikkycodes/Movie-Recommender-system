import streamlit as st
import pickle
import requests
import os


# Resolve the data files relative to this script, so the app works no matter
# which directory Streamlit is launched from (fresh clones, IDE terminals, ...).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(BASE_DIR, "movies_list.pkl")
SIM_PATH = os.path.join(BASE_DIR, "similarity.pkl")


API_KEY = "c0ccea73"   
NUM_RECS = 5           

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource(show_spinner=False)
def _load_pickles_cached(movies_path, sim_path):
    """Load the pickles once and share them. Raises on failure.

    cache_resource (not cache_data) is used because the ~800 MB similarity
    matrix would otherwise be deep-copied from the cache on every rerun, and
    because a failed load is never cached - so the app recovers by itself as
    soon as the missing file shows up.
    """
    with open(movies_path, "rb") as f:
        movies = pickle.load(f)
    with open(sim_path, "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity


def _data_path(filename):
    """Prefer the file next to app.py, else fall back to the launch directory."""
    beside_script = os.path.join(BASE_DIR, filename)
    if os.path.exists(beside_script):
        return beside_script
    in_cwd = os.path.join(os.getcwd(), filename)
    if os.path.exists(in_cwd):
        return in_cwd
    return beside_script


def load_pickles(movies_path=None, sim_path=None):
    """Load pickles and return (movies_df, similarity_matrix, error_or_None)."""
    movies_path = _data_path(movies_path or "movies_list.pkl")
    sim_path = _data_path(sim_path or "similarity.pkl")
    try:
        movies, similarity = _load_pickles_cached(movies_path, sim_path)
        return movies, similarity, None
    except FileNotFoundError as e:
        return None, None, f"Missing data file: {e.filename}"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

@st.cache_data(show_spinner=False)
def fetch_omdb_data(title):
    """Fetch OMDb data (Poster, Year, imdbRating, Plot) for a title. Cached to avoid repeated calls."""
    try:
        url = f"http://www.omdbapi.com/?t={requests.utils.requote_uri(title)}&apikey={API_KEY}"
        resp = requests.get(url, timeout=6).json()
        if resp.get("Response") == "True":
            return {  
                "Poster": resp.get("Poster", ""),
                "Year": resp.get("Year", ""),
                "imdbRating": resp.get("imdbRating", "N/A"),
                "Plot": resp.get("Plot", "")
            }
        else:
            return {"Poster": "", "Year": "", "imdbRating": "N/A", "Plot": ""}
    except Exception:
        return {"Poster": "", "Year": "", "imdbRating": "N/A", "Plot": ""}

def safe_poster_url(data):
    poster = data.get("Poster", "")
    if poster and poster != "N/A":
        return poster
    return "https://via.placeholder.com/342x513.png?text=No+Poster"

# ---------- UI ----------
st.title(" Movie Recommender System ")
st.write("Select a movie and click **Show Recommendations**. Posters, year and rating will be fetched from OMDb API.")

movies, similarity, load_err = load_pickles()
if load_err:
    st.error(f"Error loading pickles: {load_err}")
    st.caption(
        f"Searched next to `app.py` in `{BASE_DIR}` "
        f"and in the working directory `{os.getcwd()}`."
    )
    if not os.path.exists(SIM_PATH):
        st.warning(
            "`similarity.pkl` is not stored in the GitHub repo (it is ~800 MB, well over "
            "GitHub's 100 MB per-file limit), so a fresh clone will not contain it. "
            "Generate it locally, once:\n\n"
            f"```\ncd \"{BASE_DIR}\"\npython generate_similarity.py\n```\n"
            "This takes about 1-2 minutes and only needs `movies_list.pkl` + scikit-learn."
        )
    st.stop()

if not hasattr(movies, "shape") or "title" not in movies.columns:
    st.error("`movies_list.pkl` must be a pandas DataFrame with a 'title' column.")
    st.stop()
if similarity is None or len(similarity) != len(movies):
    st.error("`similarity.pkl` must be a similarity matrix matching movies length.")
    st.stop()

movies_list = movies['title'].values
select_movie = st.selectbox("Select a movie", movies_list)

if st.button("Show Recommendations"):
    with st.spinner("Finding similar movies and fetching posters..."):
        # find index
        try:
            idx = movies[movies['title'] == select_movie].index[0]
        except Exception:
            st.error("Selected movie not found in dataset.")
            st.stop()

        try:
            distances = list(enumerate(similarity[idx]))
            distances_sorted = sorted(distances, key=lambda x: x[1], reverse=True)
        except Exception as e:
            st.error(f"Error processing similarity matrix: {e}")
            st.stop()

        recs = []
        for pair in distances_sorted:
            if pair[0] == idx:
                continue
            recs.append(pair[0])
            if len(recs) >= NUM_RECS:
                break

        if not recs:
            st.warning("No recommendations found.")
            st.stop()

        rec_titles = []
        rec_posters = []
        rec_years = []
        rec_ratings = []
        rec_plots = []

        for r in recs:
            title = movies.iloc[r].title
            rec_titles.append(title)
            omdb = fetch_omdb_data(title)
            rec_posters.append(safe_poster_url(omdb))
            rec_years.append(omdb.get("Year", ""))
            rec_ratings.append(omdb.get("imdbRating", "N/A"))
            rec_plots.append(omdb.get("Plot", ""))

    cols = st.columns(len(rec_titles))
    for i, col in enumerate(cols):
        with col:
            st.image(rec_posters[i], use_container_width=True)
            st.markdown(f"**{rec_titles[i]}** ({rec_years[i]})")
            st.markdown(f"IMDb: **{rec_ratings[i]}**")
        
            with st.expander("Plot"):
                st.write(rec_plots[i])
    
            safe_link = f"https://www.imdb.com/find?q={requests.utils.requote_uri(rec_titles[i])}"
            st.markdown(f"[Open on IMDb]({safe_link})")

    st.success("Done")
