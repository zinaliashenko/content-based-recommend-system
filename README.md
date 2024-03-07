###A Content-Based Recommender System for Movies
This code implements a simple content-based recommender system algorithm for movies. The system uses textual information about movies, such as title, genres, tag lines, and reviews, to find similar movies using cosine similarity analysis.

**Running the Code:**
Run the recommendation_system.py file to get recommendations for a specific movie. Instead of "father of the bride part ii", you can use other movies.

**Functions:**
1. extract_genres
This function extracts the genres from a column in JSON format and returns them as a string.

2. select_features
Selects and processes key features for work. This includes the title, genres, tagline and overview.

3. get_recommendation
Creates recommendations based on similarity of textual information. Uses TF-IDF and cosine similarity algorithm to find the most similar movies.

**Incoming data:**
The movies_metadata.csv file contains information about various movies, including title, genres, taglines, and reviews.

**Output data:**
The result is displayed as a DataFrame containing the titles and genres of the recommended movies.

**Note:**
Make sure the path to the movies_metadata.csv file is correct for your environment. This path can be changed in the pd.read_csv function.
