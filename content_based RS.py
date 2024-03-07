import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json


# extract genres from genres_list json column
def extract_genres(genres_str):
    try:
        genres_list = json.loads(genres_str.replace("'", "\""))
        names = [genre['name'] for genre in genres_list]
        return ' '.join(names)
    except json.JSONDecodeError:
        return ''
    

def select_features(data):

    # select features to work with
    features = ['title', 'genres', 'tagline', 'overview']
    selected_features = data[features]
    selected_features = selected_features.dropna()

    selected_features[['title', 'genres', 'tagline', 'overview']] = selected_features[['title', 'genres', 'tagline', 'overview']].apply(lambda x: x.str.lower())

    # extract genres from genres_list json column
    selected_features['genres'] = selected_features['genres'].apply(extract_genres)

    # join selected features
    selected_features['combined'] = selected_features['genres'] + ' ' + selected_features['tagline'] + ' ' + selected_features['overview']

    return selected_features[['title', 'combined', 'genres']]


def get_recommendation(selected_features, movie_name):

    # use Term Frequency - Inverse Document Frequency alg to create vector matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(selected_features['combined'])

    # cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Series index-title according to the selected_features order
    indices = pd.Series(selected_features.index, index=selected_features['title']).drop_duplicates()
    # index of the wanted movie
    index = indices[movie_name]

    # list indeces - similarity to the wanted movie
    similarity_scores = list(enumerate(cosine_sim[index]))
    # sorting the list by similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # pick 10 the gratest similarity, except 1st as it is similarity to itself and is equal to 1
    similarity_scores = similarity_scores[1:11]
    # get indices for 10 similarities
    movie_indices = [i[0] for i in similarity_scores]

    # return movies names using found indices
    return selected_features[['title', 'genres']].iloc[movie_indices]


if __name__ == '__main__':

    data = pd.read_csv('movies_metadata.csv', low_memory=False)

    selected_features = select_features(data)
    
    movie_name = 'father of the bride part ii'
    result = get_recommendation(selected_features, movie_name)

    print(result)

