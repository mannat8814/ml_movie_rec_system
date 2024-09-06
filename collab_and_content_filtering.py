import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import tmdb_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

movies_df = pd.read_csv('movies.csv')  # Contains movieId, title, genres, year
ratings_df = pd.read_csv('user_reviews.csv')  # Contains userId, movieId, user_review, user_rating

#one-hot encode genres
one_hot_encoder = OneHotEncoder()
genres_encoded = one_hot_encoder.fit_transform(movies_df['genres'].str.split('|').apply(lambda x: ','.join(x)).str.get_dummies(sep=',')).toarray()
# movies_df.drop(columns=['genres'], inplace=True)
#add encoded genres to the movies dataframe
#movies_df = pd.concat([movies_df, pd.DataFrame(genres_encoded, columns=one_hot_encoder.get_feature_names_out())], axis=1)

df_id = pd.read_csv('ml-latest-small/links.csv')
df_movies = pd.read_csv('ml-latest-small/movies.csv')
df_user_ratings = pd.read_csv('ml-latest-small/movies.csv')

#merge the DataFrames on 'movie_id'
df = pd.merge(df_movies, df_id, on='movieId', how='inner')
df = df.dropna(subset = 'tmdbId')
df['tmdbId'] = df['tmdbId'].astype(int)
df = df.drop(columns='imdbId')
df['year'] = df['title'].str.extract(r'\((\d{4})\)')
df['title'] = df['title'].str.extract(r'^(.*) \(\d{4}\)$')


#one-hot encode the 'genres' column
df_genres_encoded = df['genres'].str.get_dummies(sep='|')
df = pd.concat([df, df_genres_encoded], axis=1)
df = df.drop(columns=['genres'])



def generate_user_reviews_for_movie(df):
    for mId in df['tmdbId']:
        tmdb_data.get_reviews(mId)


df_users = pd.read_csv('user_ratings.csv')
print(df_users)


#use TF-IDF vectorizer to encode user reviews
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(ratings_df['user_review']).toarray()
#add TF-IDF yo df, drop original
ratings_df = pd.concat([ratings_df, pd.DataFrame(tfidf_matrix)], axis=1)
ratings_df.drop(columns=['user_review'], inplace=True)


class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, user_id, item_id):
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        dot_product = user_emb * item_emb
        pred = self.fc(dot_product)
        return pred.squeeze()


class ContentBasedFilteringModel(nn.Module):
    def __init__(self, num_items, content_dim, embedding_dim):
        super(ContentBasedFilteringModel, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.content_fc = nn.Linear(content_dim, embedding_dim)
        
    def forward(self, item_id, content_features):
        item_emb = self.item_embedding(item_id)
        content_emb = self.content_fc(content_features)
        pred = torch.sum(item_emb * content_emb, dim=1)
        return pred

class HybridModel(nn.Module):
    def __init__(self, collab_model, content_model):
        super(HybridModel, self).__init__()
        self.collab_model = collab_model
        self.content_model = content_model
        self.fc = nn.Linear(2, 1)
    
    def forward(self, user_id, item_id, content_features):
        collab_pred = self.collab_model(user_id, item_id)
        content_pred = self.content_model(item_id, content_features)
        combined = torch.cat([collab_pred.unsqueeze(1), content_pred.unsqueeze(1)], dim=1)
        final_pred = self.fc(combined)
        return final_pred.squeeze()

#data preparation
num_users = ratings_df['userId'].nunique()
num_items = ratings_df['movieId'].nunique()
embedding_dim = 50
content_dim = genres_encoded.shape[1] + tfidf_matrix.shape[1]

#instantiate models
collab_model = CollaborativeFilteringModel(num_users, num_items, embedding_dim).to(device)
content_model = ContentBasedFilteringModel(num_items, content_dim, embedding_dim).to(device)
hybrid_model = HybridModel(collab_model, content_model).to(device)

#criterion/loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)

#training loop
epochs = 10
for epoch in range(epochs):
    hybrid_model.train()
    epoch_loss = 0
    for _, row in ratings_df.iterrows():
        user_id = torch.tensor(row['userId'], dtype=torch.long, device=device)
        item_id = torch.tensor(row['movieId'], dtype=torch.long, device=device)
        rating = torch.tensor(row['user_rating'], dtype=torch.float, device=device)
        content_features = torch.tensor(row.iloc[2:].values, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        pred = hybrid_model(user_id, item_id, content_features)
        loss = criterion(pred, rating)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(ratings_df)}')

#eval model
hybrid_model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for _, row in ratings_df.iterrows():
        user_id = torch.tensor(row['userId'], dtype=torch.long, device=device)
        item_id = torch.tensor(row['movieId'], dtype=torch.long, device=device)
        rating = torch.tensor(row['user_rating'], dtype=torch.float, device=device)
        content_features = torch.tensor(row.iloc[2:].values, dtype=torch.float, device=device)
        
        pred = hybrid_model(user_id, item_id, content_features)
        predictions.append(pred.item())
        actuals.append(rating.item())

#calculate metrics
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)

print(f'RMSE: {rmse}, MAE: {mae}')


#make recs
def recommend_movies(user_id, hybrid_model, content_features_df, n_recommendations=10):
    hybrid_model.eval()
    user_id = torch.tensor(user_id, dtype=torch.long, device=device)
    
    movie_scores = []
    with torch.no_grad():
        for movie_id, content_features in content_features_df.iterrows():
            movie_id = torch.tensor(movie_id, dtype=torch.long, device=device)
            content_features = torch.tensor(content_features.values, dtype=torch.float, device=device)
            score = hybrid_model(user_id, movie_id, content_features)
            movie_scores.append((movie_id.item(), score.item()))
    
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_scores[:n_recommendations]
    return top_movies

#test
user_id = 1  
top_movies = recommend_movies(user_id, hybrid_model, movies_df.drop(columns=['movieId', 'title', 'year']))
print(top_movies)
