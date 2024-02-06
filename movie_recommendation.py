from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample movie ratings dataset
data = {
    'user_ids': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movie_ids': [101, 102, 103, 101, 104, 102, 105, 106, 104, 107],
    'ratings': [5, 4, 3, 4, 5, 3, 2, 4, 3, 4]
}

# Create a Pandas DataFrame from the dataset
import pandas as pd
df = pd.DataFrame(data)

# Define the Reader and Dataset from surprise
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user_ids', 'movie_ids', 'ratings']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Build the collaborative filtering model using KNNBasic
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based collaborative filtering
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Make recommendations for a user
def get_movie_recommendations(user_id, n=5):
    """
    Get movie recommendations for a given user.
    """
    # Get all movie ids
    all_movie_ids = df['movie_ids'].unique()

    # Remove movies the user has already rated
    user_rated_movies = df[df['user_ids'] == user_id]['movie_ids'].values
    movies_to_predict = list(set(all_movie_ids) - set(user_rated_movies))

    # Make predictions for the user on movies not yet rated
    predicted_ratings = [model.predict(user_id, movie_id).est for movie_id in movies_to_predict]

    # Get indices of the top n recommendations
    top_indices = sorted(range(len(predicted_ratings)), key=lambda i: predicted_ratings[i], reverse=True)[:n]

    # Get the movie ids of the top recommendations
    top_movie_ids = [movies_to_predict[i] for i in top_indices]

    return top_movie_ids

# Example: Get recommendations for user 1
user_id_to_recommend = 1
recommended_movies = get_movie_recommendations(user_id_to_recommend)
print(f"Top 5 movie recommendations for user {user_id_to_recommend}: {recommended_movies}")
