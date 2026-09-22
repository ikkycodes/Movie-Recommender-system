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

# Same vectorizer settings the project used to build similarity.pkl (see
# main.ipynb / generate_similarity.py), so the on-the-fly fallback below
# produces the exact same neighbours as the pre-computed matrix.
MAX_FEATURES = 10000
STOP_WORDS = "english"


class TagsSimilarity:
    """Cosine similarity over the movie tags, computed one row at a time.

    similarity.pkl only holds the dense matrix that generate_similarity.py
    derives from movies_list.pkl['tags'] (10000 x 10000 float64 = exactly
    800 MB), which is far too big to keep in version control. This class does
    the same maths straight from the tags column but only ever materialises
    the single row that was asked for, so it needs a few MB instead of 800 MB
    and no rebuild step before the app can start.

    It exposes __len__ and __getitem__, so callers keep writing the familiar
    `similarity[idx]` and stay identical to the pre-computed matrix path.
    """

    def __init__(self, tags, max_features=MAX_FEATURES, stop_words=STOP_WORDS):
        # Imported lazily so a missing scikit-learn surfaces as a readable
        # message instead of a crash while the module is being imported.
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
        # Kept sparse: turning this into a dense array would cost ~800 MB.
        self.matrix = self.vectorizer.fit_transform([str(tag) for tag in tags])

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(self.matrix[index], self.matrix)[0]


@st.cache_resource(show_spinner="Loading movies_list.pkl...")
def _load_movies_cached(movies_path):
    """Load the movies table once and share it. Raises on failure.

    cache_resource (not cache_data) is used so the table is shared instead of
    deep-copied on every rerun, and because a failed load is never cached -
    the app recovers by itself as soon as the missing file shows up.
    """
    with open(movies_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading similarity.pkl...")
def _load_similarity_cached(sim_path):
    """Load the pre-computed matrix once and share it."""
    with open(sim_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Computing similarity from movie tags...")
def _build_tags_similarity(movies_path):
    """Build the tags-based similarity engine once and share it."""
    movies = _load_movies_cached(movies_path)
    return TagsSimilarity(movies["tags"].tolist())


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
    """Return (movies, similarity, info_or_None, error_or_None).

    similarity.pkl is used when it is present and consistent; otherwise the
    similarity matrix is computed from movies_list.pkl['tags'] on the fly, so
    the app also works from a plain clone or a GitHub Codespace where the
    800 MB pre-computed file cannot exist.
    """
    movies_path = _data_path(movies_path or "movies_list.pkl")
    sim_path = _data_path(sim_path or "similarity.pkl")

    try:
        movies = _load_movies_cached(movies_path)
    except FileNotFoundError as e:
        return None, None, None, f"Missing data file: {e.filename}"
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"

    if not hasattr(movies, "columns") or "title" not in movies.columns:
        return None, None, None, "`movies_list.pkl` must be a pandas DataFrame with a 'title' column."

    # The search below indexes the matrix by position (`similarity[idx]`) and
    # reads the results back with iloc, so a non-default index would silently
    # return the wrong movies. Normalise it before doing any position maths.
    if movies.index.tolist() != list(range(len(movies))):
        movies = movies.reset_index(drop=True)

    note = None
    if os.path.exists(sim_path):
        try:
            similarity = _load_similarity_cached(sim_path)
            if len(similarity) == len(movies):
                return movies, similarity, None, None
            note = (
                f"`similarity.pkl` holds {len(similarity)} rows but there are "
                f"{len(movies)} movies, so it was ignored."
            )
        except Exception as e:
            note = f"`similarity.pkl` could not be read ({type(e).__name__}: {e}), so it was ignored."

    if "tags" not in movies.columns:
        prefix = f"{note} " if note else ""
        return None, None, None, (
            f"{prefix}There is no usable `similarity.pkl` and `movies_list.pkl` has no "
            "'tags' column to rebuild it from."
        )

    try:
        similarity = _build_tags_similarity(movies_path)
    except ImportError as e:
        return None, None, None, (
            f"{e} - install the dependencies with `pip install -r requirment.txt`."
        )
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"

    info = (
        "`similarity.pkl` is not present (it is ~800 MB, so it is not kept in the "
        "repository), so similarities are being computed from the `tags` column with "
        "the same CountVectorizer and cosine similarity settings, one row at a time. "
        "No large file is required. Run `python generate_similarity.py` if you would "
        "rather cache the full matrix to disk."
    )
    if note:
        info = f"{note} {info}"
    return movies, similarity, info, None

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

movies, similarity, data_info, load_err = load_pickles()
if load_err:
    st.error(f"Error loading data: {load_err}")
    st.caption(
        f"Searched next to `app.py` in `{BASE_DIR}` "
        f"and in the working directory `{os.getcwd()}`."
    )
    if not os.path.exists(MOVIES_PATH):
        st.warning(
            "`movies_list.pkl` is missing. It is committed to the repository, so pull "
            "the latest `main` (or re-clone) and start the app again."
        )
    st.stop()

if data_info:
    st.info(data_info)

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
