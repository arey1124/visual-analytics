from collections import Counter
import os
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from threading import Thread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from plotly.subplots import make_subplots




# Arihant
spotify_data = pd.read_csv('final_datasets/result_spotify_dataset.csv')
genre_data = pd.read_csv('final_datasets/train.csv')

def mapper(col):
    coded_dict = dict()
    cter = 1
    encoded = []
    
    for val in spotify_data[col]:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1
        
        encoded.append(coded_dict[val])
    return encoded

artist_id = mapper('artistname')
user_ids = mapper('user_id')

def normalize(numerical_columns, df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df.loc[:, numerical_columns] = min_max_scaler.fit_transform(df[numerical_columns])
    return df

# Handle spotify dataset (Preprocessing) 
spotify_data['artistname_encoded'] = artist_id
spotify_data['user_id_encoded'] = user_ids
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# Handle genre dataset (Preprocessing)
genre_data.dropna(inplace=True)
genre_data.drop(['Popularity', 'time_signature', 'duration_in min/ms'], axis='columns', inplace=True)
genre_data.rename(columns={'Artist Name': 'artistname', 'Track Name': 'trackname'}, inplace=True)
# Normalize numerical columns
columns = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"]
genre_data = normalize(columns, genre_data)

# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

genre_identification_model = None
genre_identification_preprocessor = None

@app.callback(
    Output("tempo-valence-figure", "figure"),
    [
        Input("user-tempo-valence-analysis-dropdown", "value"),
        Input('user-tempo-valence-analysis-playlist-dropdown', 'value')
    ]
)
def tempo_valence_figure(user_id, playlist_name):
    numerical_columns = ['valence', 'tempo']
    df = spotify_data[(spotify_data['playlistname'] == playlist_name) & (spotify_data['user_id']==user_id)]
    df = normalize(numerical_columns, df)
    df = df[numerical_columns]
    try:
        # Create scatter plot
        fig = px.scatter(df, x="valence", y="tempo", color="valence")
    except ValueError:
        fig = px.scatter(df, x="valence", y="tempo", color="valence")
    return fig

@app.callback(
    Output("song-duration-figure", "figure"),
    [
        Input("user-song-duration-dropdown", "value"),
        Input('user-song-duration-playlist-dropdown', 'value')
    ]
)
def song_duration_figure(user_id, playlist_name):
    df = spotify_data[(spotify_data['playlistname'] == playlist_name) & (spotify_data['user_id'] == user_id)]

    # Convert duration_ms to minutes
    df['duration_minutes'] = df['duration_ms'] / (1000 * 60)
    # Calculate average duration
    avg_duration = df['duration_minutes'].mean()
    sorted_df = df.sort_values(by='duration_minutes', ascending=False)
    sorted_df['trackname'] = sorted_df['trackname'].apply(lambda x: x[:25] if len(x) > 25 else x)
    try:
        fig = px.histogram(sorted_df, x='trackname', y='duration_minutes')
    except ValueError:
        fig = px.histogram(sorted_df, x='trackname', y='duration_minutes')

    fig.add_shape(
        type='line',
        x0=0,
        x1=len(df['trackname']) - 1,
        y0=avg_duration,
        y1=avg_duration,
        line=dict(color='red', width=2, dash='dash')
    )

    fig.update_layout(
        xaxis=dict(title='Song Name'),
        yaxis=dict(title='Duration (minutes)'),
        showlegend=False
    )
    return fig

@app.callback(
    Output("feature-distribution-figure", "figure"),
    [
        Input("graph-toggle", "value"),
        Input("feature-distribution-radio", "value")
    ]
)
def feature_distribution_figure(graph_type, feature):
    if graph_type == 'Violin':
        fig = px.violin(spotify_data, y=feature.lower())
    else:
        fig = px.histogram(spotify_data, x=feature.lower(), nbins=20)
    return fig

@app.callback(
    Output("feature-correlation-heatmap-figure", "figure"),
    [
        Input("feature-correlation-heatmap-figure", "id")   
    ],  # dummy Input
)
def create_heatmap(_):
    numerical_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    correlation = spotify_data[numerical_columns].corr()
    heatmap = px.imshow(correlation, color_continuous_scale='sunset', aspect="auto")
    heatmap.update_traces(text=np.around(correlation.values, decimals=2), texttemplate="%{text}")
    heatmap.update_xaxes(side="top")
    return heatmap

@app.callback(
    Output("user-song-preference-analysis-figure", "figure"),
    [
        Input("user-song-preference-analysis-dropdown", "value"),
        Input('user-song-preference-analysis-radio', 'value'),
        Input("user-song-preference-analysis-playlist-dropdown", "value")
    ]
)
def create_radar_chart(user_id, radio_value, playlist_name=None):
    numerical_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    if radio_value == 'Average':
        df = spotify_data[spotify_data['user_id']==user_id]
        df = normalize(numerical_columns, df)
        df = df[numerical_columns]
        average_features = df.mean().tolist()
        fig = px.line_polar(r=average_features, theta=numerical_columns, line_close=True)
    elif radio_value == 'Each song':
        df = spotify_data[spotify_data['user_id']==user_id]
        df = normalize(numerical_columns, df)
        df = df.reset_index().melt(id_vars='trackname', value_vars=numerical_columns)
        fig = px.line_polar(df, r='value', theta='variable', color='trackname', line_close=True)
    elif radio_value in ['Each playlist', 'Average in a playlist', 'Each song in a playlist']:
        if radio_value == 'Each playlist':
            df = spotify_data[spotify_data['user_id']==user_id]
            df = normalize(numerical_columns, df)
            df = df.groupby('playlistname')[numerical_columns].mean().reset_index()
            df = df.reset_index().melt(id_vars='playlistname', value_vars=numerical_columns)
            fig = px.line_polar(df, r='value', theta='variable', color='playlistname', line_close=True)
        elif radio_value == 'Average in a playlist':
            df = spotify_data[(spotify_data['playlistname'] == playlist_name) & (spotify_data['user_id']==user_id)]
            df = normalize(numerical_columns, df)
            df = df[numerical_columns]
            average_features = df.mean().tolist()
            fig = px.line_polar(r=average_features, theta=numerical_columns, line_close=True)
        elif radio_value == 'Each song in a playlist':
            df = spotify_data[(spotify_data['playlistname'] == playlist_name) & (spotify_data['user_id']==user_id)]
            df = normalize(numerical_columns, df)
            df = df.reset_index().melt(id_vars='trackname', value_vars=numerical_columns)
            fig = px.line_polar(df, r='value', theta='variable', color='trackname', line_close=True)

    fig.update_traces(fill='toself')
    return fig

def create_user_artist_matrix(spotify_data):
    spotify_data['interact'] = 1
    user_artist = spotify_data.groupby(['user_id_encoded', 'artistname_encoded'])['interact'].max().unstack().fillna(0).astype('int')
    return user_artist

def generate_recommendations_collaborative(user_id, user_artist, threshold):
    user_id_encoded = spotify_data[spotify_data['user_id']==user_id]['user_id_encoded'].values[0]
    # compute similarity of each user to the provided user
    similarity_users = np.dot(user_artist.loc[user_id_encoded], user_artist.transpose())
    similarity_users = pd.DataFrame(similarity_users, columns=['similarity'])
    
    # create similar_user_id column
    # Adding 1 because index starts from 0 in pandas
    similarity_users['similar_user_id'] = similarity_users.index + 1
    
    # Find common artists to apply threshold
    common_artists = find_common_artists(user_id, user_artist)
    
    # Apply threshold
    similarity_users = similarity_users[similarity_users['similar_user_id'].isin([user for user, count in common_artists.items() if count >= threshold])]
 
    # calculate the percentage of similarity
    similarity_users['similarity_percentage'] = similarity_users['similar_user_id'].apply(lambda x: jaccard_similarity(user_id_encoded, x, user_artist))
    
    # remove the own user's id
    similarity_users = similarity_users[similarity_users['similar_user_id'] != user_id_encoded]
    
    # sort by similarity and then by percentage similarity
    similarity_users.sort_values(by=['similarity', 'similarity_percentage'], ascending=False, inplace=True)
    
    # Remove recommendations that have no similarity
    similarity_users = similarity_users[similarity_users['similarity_percentage'] > 0]
    
    # Get the list of similar user ids for evaluating the algorithm
    recommendations = similarity_users['similar_user_id'].tolist()
    
    precision, recall, f1_score = calculate_metrics(recommendations, common_artists, threshold=5)
    
    similarity_users = similarity_users.assign(similar_user_id=similarity_users['similar_user_id'].apply(lambda x: spotify_data.loc[spotify_data['user_id_encoded'] == x, 'user_id'].values[0] if x in spotify_data['user_id_encoded'].values else x))
    col = similarity_users.pop('similar_user_id')
    similarity_users.insert(0, col.name, col)
    
    return similarity_users, precision, recall, f1_score

def jaccard_similarity(user1, user2, user_artist):
    # print(user1, user2)
    # Get the artists for each user
    artists_user1 = set(user_artist.loc[user1].to_numpy().nonzero()[0])
    artists_user2 = set(user_artist.loc[user2].to_numpy().nonzero()[0])
    # print(artists_user1)
    
    
    # Calculate the intersection and union of the artists
    intersection = len(artists_user1.intersection(artists_user2))
    union = len(artists_user1.union(artists_user2))
    
    # Calculate Jaccard Similarity
    similarity = intersection / union if union != 0 else 0
    
    return similarity * 100

def find_common_artists(user_id, user_artist):
    user_id_encoded = spotify_data[spotify_data['user_id']==user_id]['user_id_encoded'].values[0]
    # Find the artists that the target user has interacted with
    target_artists = set(user_artist.loc[user_id_encoded][user_artist.loc[user_id_encoded]==1].index)

    # Create a dictionary to store each user and their count of common artists with the target user
    common_artists = {}

    # Iterate over each user
    for user in user_artist.index:
        if user != user_id:  # Exclude the target user
            # Find the artists that the current user has interacted with
            user_artists = set(user_artist.loc[user][user_artist.loc[user]==1].index)
            
            # Find the common artists between the target user and the current user
            common = target_artists & user_artists
            
            # Store the count of common artists in the dictionary
            common_artists[user] = len(common)

    return common_artists

def calculate_metrics(recommendations, common_artists, threshold):
    # Calculate the number of recommended users
    num_recommended = len(recommendations)

    # Calculate the number of relevant users (i.e., users who have more than 'threshold' artists in common with the target user)
    num_relevant = sum(1 for user, count in common_artists.items() if count >= threshold)

    # Calculate the number of true positives (i.e., relevant users who were also recommended)
    num_true_positives = sum(1 for user in recommendations if common_artists[user] >= threshold)

    # Precision is the proportion of recommended users that are relevant
    precision = num_true_positives / num_recommended if num_recommended else 0

    # Recall is the proportion of relevant users that are recommended
    recall = num_true_positives / num_relevant if num_relevant else 0

    # F1 score is the harmonic mean of precision and recall
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1_score

def calculate_threshold(user_id):
    user_id_encoded = spotify_data[spotify_data['user_id']==user_id]['user_id_encoded'].values[0]
    user_artist = create_user_artist_matrix(spotify_data)
    threshold = 0
    user_artist_count = len(user_artist.loc[user_id_encoded][user_artist.loc[user_id_encoded]==1])
    for i in range(0, user_artist_count+1):
        user_recommendations, precision, recall, f1_score = generate_recommendations_collaborative(user_id, user_artist, threshold=threshold)
        if len(user_recommendations) == 0:
            return threshold
        threshold+=1

# Callback to update the graph based on the selected threshold
def metrics_thresholds_graph_collaborative(user_id, user_artist):
    collaborative_precision_list = []
    collaborative_recall_list = []
    collaborative_f1_score_list = []

    for t in range(0, 11):
        user_recommendations, collaborative_precision, collaborative_recall, collaborative_f1_score = generate_recommendations_collaborative(user_id=user_id, user_artist=user_artist, threshold=t)
        collaborative_precision_list.append(collaborative_precision)
        collaborative_recall_list.append(collaborative_recall)
        collaborative_f1_score_list.append(collaborative_f1_score)

    thresholds = list(range(0, 11, 1))

    df = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': collaborative_precision_list,
        'Recall': collaborative_recall_list,
        'F1 Score': collaborative_f1_score_list
    })

    # Plot with Plotly Express
    fig = px.line(df, x='Threshold', y=['Precision', 'Recall', 'F1 Score'], markers=True, title='Evaluation Metrics at Different Thresholds')
    fig.update_xaxes(title='Threshold')
    fig.update_yaxes(title='Score')
    fig.update_layout(legend=dict(title='Metrics'))
    
    return fig

@app.callback(
    [
        Output('user-song-preference-analysis-playlist-dropdown', 'options'),
        Output('user-song-preference-analysis-playlist-dropdown', 'value')
    ],
    [
        Input('user-song-preference-analysis-dropdown', 'value'), 
        Input('user-song-preference-analysis-radio', 'value')
    ]
)
def update_dropdown_song_preference(user_id, radio_value):
    if radio_value in ['Average in a playlist', 'Each song in a playlist']:
        df = spotify_data[spotify_data['user_id'] == user_id]
        options = [{'label': playlist, 'value': playlist} for playlist in df['playlistname'].unique()]
        return options, options[0]['value'] 
    else:
        return [{'label':'default', 'value':'default', 'disabled':True}], 'default'

@app.callback(
    [
        Output('similar-user-matching-content-based-slider-div', 'children'),
        Output('similar-user-matching-content-based-slider-div', 'style')
    ],
    [
        Input('similar-user-matching-dropdown', 'value'),
        Input('similar-user-matching-radio', 'value')
    ]
)
def update_dropdown_content_based(user_id, radio_value):
    user_features = spotify_data[spotify_data['user_id_encoded'] == 100].mean(numeric_only=True)[:-3].to_dict()
    div = html.Div(children=[])    
    div.children.append(html.Div([
        html.H4(children="Adjust threshold (Number of features with values equal or more than your values.)"),
        dcc.Slider(id="similar-user-matching-content-based-slider", min=0, max=13, step=1, value=4)
    ]))
    if radio_value == "Your specific music taste":
        return div, {'display': 'block', 'margin-top': '20px', 'margin-bottom' : '20px'}
    else:
        return div, {'display': 'none'}

@app.callback(
    [
        Output('user-tempo-valence-analysis-playlist-dropdown', 'options'),
        Output('user-tempo-valence-analysis-playlist-dropdown', 'value')
    ],
    [
        Input('user-tempo-valence-analysis-dropdown', 'value'),
    ]
)
def update_tempo_valence_dropdown(user_id):
    df = spotify_data[spotify_data['user_id'] == user_id]
    options = [{'label': playlist, 'value': playlist} for playlist in df['playlistname'].unique()]
    return options, options[0]['value']

@app.callback(
    [
        Output('user-song-duration-playlist-dropdown', 'options'),
        Output('user-song-duration-playlist-dropdown', 'value')
    ],
    [
        Input('user-song-duration-dropdown', 'value'),
    ]
)
def update_song_duration_dropdown(user_id):
    df = spotify_data[spotify_data['user_id'] == user_id]
    options = [{'label': playlist, 'value': playlist} for playlist in df['playlistname'].unique()]
    return options, options[0]['value']   

@app.callback(
    Output('similar-user-matching-collaborative-slider', 'max'),
    Input('similar-user-matching-dropdown', 'value')
)
def update_slider_max(selected_user):
    max_threshold = calculate_threshold(selected_user)
    return max_threshold

@app.callback(
    Output('similar-user-matching-collaborative-slider-div', 'style'),
    Input('similar-user-matching-radio', 'value')
)
def update_slider_visibility(selected_value):
    if selected_value == 'Artists you have in common':
        return {'display': 'block', 'margin-top': '20px', 'margin-bottom' : '20px'}
    else:
        return {'display': 'none'}

def generate_recommendations_content_based(user_id, spotify_data, threshold):
    user_id_encoded = spotify_data[spotify_data['user_id']==user_id]['user_id_encoded'].values[0]
    # Get user feature values
    feature_values_dict = spotify_data[spotify_data['user_id_encoded'] == user_id_encoded].mean(numeric_only=True)[:-3].to_dict()
    
    # Aggregate spotify_data by user
    data_agg = spotify_data[feature_values_dict.keys()].groupby(spotify_data['user_id_encoded']).mean()
    
    # print(data_agg, feature_values_dict)
    
    # Adjust the feature values of the target user
    data_agg.loc[user_id_encoded] = list(feature_values_dict.values())
    
    # Calculate the cosine similarity between the user and all other users
    similarity_scores = cosine_similarity(data_agg.loc[user_id_encoded].values.reshape(1, -1), data_agg.values)[0]
    
    # Create a DataFrame for the similarity scores
    similarity_df = pd.DataFrame(similarity_scores, columns=['similarity'], index=data_agg.index)
    
    similarity_df.reset_index(inplace=True)

    # Remove the user's own id
    similarity_df = similarity_df[similarity_df['user_id_encoded'] != user_id_encoded]
    
    # Sort the users by similarity score
    similarity_df.sort_values(by='similarity', ascending=False, inplace=True)
    
    # Calculate percentages of similarity
    similarity_df['similarity_percentage'] = similarity_df['similarity'] * 100
    
    recommended_users = similarity_df['user_id_encoded'].tolist()
    
    precision, recall, f1_score = calculate_metrics_content_based(recommended_users, spotify_data, feature_values_dict, threshold)
    
    # similarity_df = similarity_df.assign(user_id_encoded=similarity_df['user_id_encoded'].apply(lambda x: spotify_data.loc[spotify_data['user_id_encoded'] == x, 'user_id'].values[0] if x in spotify_data['user_id_encoded'].values else x))
    
    # similarity_df = similarity_df.rename(columns={'user_id_encoded':'similar_user_id'})
    
    return similarity_df, precision, recall, f1_score


def calculate_metrics_content_based(recommended_users, data, user_features, threshold):
    tp = 0
    fp = 0
    fn = 0
    relevant_users = set(data[data['user_id_encoded'].isin(recommended_users)]['user_id_encoded'])

    for user in recommended_users:
        # Calculate the aggregate of their song features
        similar_user_features = data[data['user_id_encoded'] == user].mean(numeric_only=True)[:-3].to_dict()

        # Compare that to the user_features
        num_similar_features_above_threshold = sum(similar_user_features[feature] >= user_features[feature] for feature in user_features)

        is_relevant = num_similar_features_above_threshold >= threshold

        if is_relevant:
            tp += 1
        else:
            # If user is recommended but not relevant
            if user in relevant_users:
                fp += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def metrics_thresholds_graph_content_based(user_id):
    content_based_precision_list = []
    content_based_recall_list = []
    content_based_f1_score_list = []

    for t in np.arange(1, 13, 1):
        cosine_sim_df, content_based_precision, content_based_recall, content_based_f1_score = generate_recommendations_content_based(user_id, spotify_data, t)
        content_based_precision_list.append(content_based_precision)
        content_based_recall_list.append(content_based_recall)
        content_based_f1_score_list.append(content_based_f1_score)

    thresholds = list(np.arange(1, 13, 1))

    df = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': content_based_precision_list,
        'Recall': content_based_recall_list,
        'F1 Score': content_based_f1_score_list
    })

    # Plot with Plotly Express
    fig = px.line(df, x='Threshold', y=['Precision', 'Recall', 'F1 Score'], markers=True, title='2. Evaluation Metrics at Different Thresholds')
    fig.update_xaxes(title='Threshold')
    fig.update_yaxes(title='Score')
    fig.update_layout(legend=dict(title='Metrics'))

    return fig

@app.callback(
    [
        Output("similar-user-matching-figure1", "figure"),
        Output("similar-user-matching-figure2", "figure"),
        Output("similar-user-matching-table", "children")
    ], 
    [
        Input("similar-user-matching-dropdown", "value"),
        Input("similar-user-matching-radio", "value"),
        Input("similar-user-matching-collaborative-slider","value"),
        Input("similar-user-matching-content-based-slider","value"),
        
        # *[Input(f'{feature}-slider', 'value') for feature in features]
    ]
)
def similar_user_matching(user_id, radio_value, collaborative_threshold, content_based_threshold):
    if radio_value == "Artists you have in common":
        user_artist = create_user_artist_matrix(spotify_data)
        similarity_df, precision, recall, f1_score = generate_recommendations_collaborative(user_id, user_artist, collaborative_threshold)
        fig2 = metrics_thresholds_graph_collaborative(user_id, user_artist)
    elif radio_value == "Your specific music taste":
        similarity_df, precision, recall, f1_score = generate_recommendations_content_based(user_id, spotify_data, content_based_threshold)
        fig2 = metrics_thresholds_graph_content_based(user_id)
    else: return None, None, None

    df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Score': [precision, recall, f1_score]
    })
    # Create the plot
    fig1 = px.bar(df, x='Metric', y='Score', title='1. Evaluation Metrics')
    
    # Create the table
    table = dash_table.DataTable(
        data=similarity_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in similarity_df.columns],
        page_action='native',
        page_current=0,
        page_size=5,
        # Center the table on the page
        style_table={'margin-left': 'auto', 'margin-right': 'auto'}
        
    )
    return fig1, fig2, table

@app.callback(
    Output('song-clustering-figure', 'figure'),
    Input('song-clustering-features-dropdown', 'value'),
    Input('song-clustering-slider', 'value')
)
def song_clustering_exploration(features, clusters):
    if not features or len(features) < 2:
        return dash.no_update
    
    X = spotify_data[features]
    # Normalize the values of each column
    X = (X - X.mean()) / X.std()

    # Initialize the KMeans model with 8 clusters
    kmeans = KMeans(n_clusters=clusters, random_state=1, n_init=10)

    # Fit the model to the spotify_data
    kmeans.fit(X) 
    clusters = kmeans.predict(X)

    pca = PCA(n_components=2)  # Initialize a PCA model with 2 components
    df_2d = pca.fit_transform(X)  # Reduce the spotify_data to two dimensions using the PCA model

    # Plot the spotify_data points on a scatter plot
    # Coloring the spotify_data points according to their cluster assignment
    fig = px.scatter(
        spotify_data,
        x=df_2d[:, 0],
        y=df_2d[:, 1],
        color=clusters,
        hover_data={'trackname': True},
        title='Clustering on Audio Features'
    )
    fig.update_layout(
        width=800,  # Set width in pixels
        height=600,
        xaxis_title='X-axis', 
        yaxis_title='Y-axis'
    )

    return fig

@app.callback(
    Output('genre-identification-figure1', 'figure'),
    Output('genre-identification-figure2', 'figure'),
    Input('genre-identification-learning-rate-slider', 'value')
)
def train_and_evaluate_model(alpha):
    global genre_identification_model
    global genre_identification_preprocessor
    # Split genre_data into features (X) and target labels (y)
    X, y = genre_data.drop('Class', axis=1), genre_data['Class']

    # Split genre_data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # One-hot encode categorical columns
    categorical_cols = ['artistname', 'trackname']
    genre_identification_preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
    X_train = genre_identification_preprocessor.fit_transform(X_train)
    X_test = genre_identification_preprocessor.transform(X_test)

    # Encode target labels
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.todense(), y_train_encoded))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.todense(), y_test_encoded))
    train_dataset = train_dataset.shuffle(X_train.shape[0]).batch(32)
    test_dataset = test_dataset.batch(32)

    # Build and compile the model
    input_shape = X_train.shape[1]
    genre_identification_model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.7),
        Dense(11, activation='softmax')
    ])
    genre_identification_model.compile(optimizer=Adam(learning_rate=alpha), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = genre_identification_model.fit(train_dataset, epochs=10, batch_size=32, validation_data=test_dataset)

    # Evaluate the model on test data
    test_loss, test_accuracy = genre_identification_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy}")

    # Plot loss and accuracy curves using Plotly Express
    evaluation_fig1 = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy Curve', 'Loss Curve'))

    evaluation_fig1.add_trace(go.Scatter(x=list(range(1, 11)), y=history.history['accuracy'], mode='lines', name='accuracy'),
                  row=1, col=1)
    evaluation_fig1.add_trace(go.Scatter(x=list(range(1, 11)), y=history.history['val_accuracy'], mode='lines', name='val_accuracy'),
                  row=1, col=1)
    evaluation_fig1.update_xaxes(title_text='Epoch', row=1, col=1)
    evaluation_fig1.update_yaxes(title_text='Accuracy', row=1, col=1)

    evaluation_fig1.add_trace(go.Scatter(x=list(range(1, 11)), y=history.history['loss'], mode='lines', name='loss'), row=1, col=2)
    evaluation_fig1.add_trace(go.Scatter(x=list(range(1, 11)), y=history.history['val_loss'], mode='lines', name='val_loss'),
                  row=1, col=2)
    evaluation_fig1.update_xaxes(title_text='Epoch', row=1, col=2)
    evaluation_fig1.update_yaxes(title_text='Loss', row=1, col=2)

    # Plot confusion matrix using Plotly Express
    y_test_pred = genre_identification_model.predict(test_dataset)
    cm = confusion_matrix(np.argmax(y_test_encoded, axis=1), np.argmax(y_test_pred, axis=1))

    labels = sorted(y.unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Assuming cm_df is the DataFrame containing the confusion matrix
    evaluation_fig2 = px.imshow(
        cm_df,
        labels=dict(x="Predicted", y="True"),
        x=labels,
        y=labels,
        color_continuous_scale='Agsunset'
    )
    
    evaluation_fig2.update_traces(text=cm_df.values, texttemplate="%{text}")

    # Update the color bar title
    evaluation_fig2.update_layout(coloraxis_colorbar=dict(title="Count"))


    return evaluation_fig1, evaluation_fig2

# New callback to update the store whenever genre_identification_model is updated
@app.callback(
    Output('genre-identification-store', 'data'),
    Input('genre-identification-figure1', 'figure')
)
def update_store_on_model_change(figure):
    global genre_identification_model
    model_json = genre_identification_model.to_json()
    return model_json

@app.callback(
    Output('genre-identification-figure3', 'figure'),
    Output('genre-identification-figure4', 'figure'),
    Output('genre-identification-most-listened-genre', 'children'),
    [Input('genre-identification-store', 'data'),
    Input('genre-identification-dropdown', 'value')]
)
def identify_genre(model_json, user_id):
    columns = ["danceability",	"energy", "key",	"loudness",	"mode",	"speechiness",	"acousticness",	"instrumentalness",	"liveness",	"valence",	"tempo"]
    if model_json == None:
        return {}, {}, "Please wait till the model gets trained!"
    
    model = model_from_json(model_json)
    
    user_data = spotify_data[spotify_data['user_id']==user_id]
    # Prepare the user data

    # user_data = user_data.to_frame().T  # Convert user_data to a DataFrame with a single row

    labels = ["Rock", "Indie", "Alt", "Pop", "Metal", "HipHop", "Alt_Music", "Blues", "Acoustic/Folk", "Instrumental", "Country"]

    user_data = normalize(columns, user_data)
        
    user_data = genre_identification_preprocessor.transform(user_data)

    # Make predictions using the loaded model
    user_predictions = model.predict(user_data)

    # Decode the predictions to get the predicted genre
    predicted_genre_index = np.argmax(user_predictions, axis=1)

    predicted_genre_labels = [labels[i] for i in predicted_genre_index]

    # Find the most frequent genre
    most_frequent_genre = Counter(predicted_genre_labels).most_common(1)[0][0]

    # Count the occurrences of each genre
    genre_counts = Counter(predicted_genre_labels)

    # Ensure that all labels are present in the counts, even if some have zero occurrences
    df = pd.DataFrame({'Genre': labels, 'Count': [genre_counts[label] for label in labels]})

    # Sort the DataFrame by count in descending order
    df = df.sort_values(by='Count', ascending=False)

    fig3 = px.bar(df, x='Genre', y='Count', color='Genre', title='3. User Overall Genre Preference',
                labels={'Count': 'Genre Count'}, color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Line Polar Chart
    df = spotify_data[spotify_data['user_id']==user_id]
    df = normalize(columns, df)
    df = df[columns]
    average_features = df.mean().tolist()
    fig4 = px.line_polar(r=average_features, theta=columns, line_close=True, title='4. User Aggregated Genre Preference')
    fig4.update_traces(fill='toself')
    
    return fig3, fig4, most_frequent_genre

# sliders = [html.Div(
#                 [html.Label(feature),
#                 dcc.Slider(
#                     id=f'{feature}-slider',
#                     min=0,
#                     max=1,
#                     step=0.05,
#                     value=0
#                 )]
#         ) for feature in features]    

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Harmonizing Hearts", className="display-6"),
        dbc.Nav(
            [
                dbc.NavLink("Analysis & Exploration", href="/", active="exact"),
                dbc.NavLink("User Song Preferences Analysis", href="/user-song-preference", active="exact"),
                dbc.NavLink("Similar User Matching", href="/similar-user-matching", active="exact"),
                dbc.NavLink("Song Clustering & Exploration", href="/song-clustering", active="exact"),
                dbc.NavLink("Genre Identification", href="/genre-identification", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    html.Div(
        [
            dcc.Location(id="url"), 
            sidebar, 
            content,
            dcc.Store(id='genre-identification-store', storage_type='memory', data=genre_identification_model),
        ]
    ),
])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return dbc.Container([
            html.Div([
                html.H2('Tempo & Valence Analysis', className='mt-4 mb-4'),  # Add margin top & bottom
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4('Select user', className='mb-2'),  # Add margin bottom
                                dcc.Dropdown(
                                    id='user-tempo-valence-analysis-dropdown',
                                    options=[{'label': user, 'value': user} for user in spotify_data['user_id'].unique()],
                                    value=spotify_data['user_id'][0],
                                ),
                            ],
                            md=6,  # Specify column size for medium screens
                        ),
                        dbc.Col(
                            [
                                html.H4('Select playlist', className='mb-2'),  # Add margin bottom
                                dcc.Dropdown(
                                    id='user-tempo-valence-analysis-playlist-dropdown',
                                    options=[{'label': 'default', 'value': 'default', 'disabled': True}],
                                ),
                            ],
                            md=6,  # Specify column size for medium screens
                        ),
                    ],
                    className='mb-4',  # Add margin bottom to the row
                ),
                dcc.Graph(
                    id='tempo-valence-figure',
                    className='mb-4',  # Add margin bottom to the graph
                ),
                html.P('Note:'),
                html.Br(),
                html.P('The scatter plot visually represents songs according to their tempo and emotional valence, offering insights into emotional content based on position: higher tempo values imply faster-paced songs, while higher valence values indicate more positive emotions in music.')
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
            html.Div([
                html.H2('Song Duration Analysis', className='mt-4 mb-4'),  # Add margin top & bottom
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4('Select user', className='mb-2'),  # Add margin bottom
                                dcc.Dropdown(
                                    id='user-song-duration-dropdown',
                                    options=[{'label': user, 'value': user} for user in spotify_data['user_id'].unique()],
                                    value=spotify_data['user_id'][0],
                                ),
                            ],
                            md=6,  # Specify column size for medium screens
                        ),
                        dbc.Col(
                            [
                                html.H4('Select playlist', className='mb-2'),  # Add margin bottom
                                dcc.Dropdown(
                                    id='user-song-duration-playlist-dropdown',
                                    options=[{'label': 'default', 'value': 'default', 'disabled': True}],
                                ),
                            ],
                            md=6,  # Specify column size for medium screens
                        ),
                    ],
                    className='mb-4',  # Add margin bottom to the row
                ),
                dcc.Graph(
                    id='song-duration-figure',
                    className='mb-4',  # Add margin bottom to the graph
                ),
                html.P('Note:'),
                html.Br(),
                html.P('The bar plot showcases the distribution of favored song lengths, helping users understand their engagement patterns with songs of various durations, thereby allowing them to recognize their preference for specific song durations for a personalized music experience.')
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
            html.Div([
                html.H2('Feature Distribution Plots'),
                dbc.RadioItems(
                    id="graph-toggle",
                    options=[
                        {'label': 'Violin Plot', 'value': 'Violin'},
                        {'label': 'Histogram', 'value': 'Histogram'}
                    ],
                    value='Violin',
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-secondary",
                    labelCheckedClassName="active",
                    style={'margin-top': '20px', 'margin-bottom' : '20px'}
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id='feature-distribution-figure'),
                        width={'size': 8},  # Width for medium-sized screens
                        className='mb-4',
                    ),
                    dbc.Col(
                        dcc.RadioItems(
                            id="feature-distribution-radio",
                            options=[
                                {'label': i, 'value': i} for i in [
                                    'Danceability', 'Energy', 'Loudness', 'Speechiness',
                                    'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo'
                                ]
                            ],
                            value='Danceability',
                            inputStyle={'margin-right': '10px'},  # Adjust radio button margin for better spacing
                            labelStyle={'display': 'block', 'background-color': '#f2f2f2', 'padding': '5px 10px', 'border-radius': '5px'},
                            # Styling selected radio button
                            inputClassName='radio-selected',
                            style= {'margin-top': '60px'}
                        ),
                        width={'size': 4, 'order': 'last'}  # Width and order for medium-sized screens
                    ),
                ]),
                html.P('Note:'),
                html.Br(),
                html.P('A violin plot displays the distribution of music features like danceability, energy, and loudness, showcasing their spread and concentration. Additionally, the inclusion of a toggle button allows users to switch between violin plots and histograms, providing versatile views of feature distributions for a better understanding of their music preferences.')
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
            html.Div([
                html.H2('Feature Correlation Heatmap'),
                dcc.Graph(
                    id='feature-correlation-heatmap-figure',
                    className='mb-4',
                ),
                html.P('Note:'),
                html.Br(),
                html.P('A heatmap visually represents correlations between different features by using colors to indicate the strength of relationships. For example, if two features have a positive correlation, they will appear as a darker color, while a negative correlation will be shown as a lighter color. In the context of music features like loudness and energy, a heatmap could display how closely related these attributes are, helping users understand how these characteristics co-vary or differ in their preferred songs')
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
        ])
    elif pathname == "/user-song-preference" :
        return dbc.Container(
            html.Div([
            html.H2('User Song Preferences Analysis', className='mt-4 mb-4'),  # Heading with margins
                html.Div(
                    [
                        html.H4('Select user:'),
                        dcc.Dropdown(
                            id="user-song-preference-analysis-dropdown",
                            options=[{'label': user, 'value': user} for user in spotify_data['user_id'].unique()],
                            value=spotify_data['user_id'][0],
                        ),
                    ],
                    style={'margin-bottom': '20px'},  # Add margin bottom
                ),
                html.Div(
                    [
                        html.H4('View chart for:'),
                        dbc.RadioItems(
                            id="user-song-preference-analysis-radio",
                            options=[
                                {'label': 'Average', 'value': 'Average'},
                                {'label': 'Each song', 'value': 'Each song'},
                                {'label': 'Each playlist', 'value': 'Each playlist'},
                                {'label': 'Average in a playlist', 'value': 'Average in a playlist'},
                                {'label': 'Each song in a playlist', 'value': 'Each song in a playlist'},
                            ],
                            value='Average',
                            style={'display' : 'flex'},
                            inputClassName="btn-check",
                            labelClassName="btn btn-outline-secondary",
                            labelCheckedClassName="active",
                        ),
                    ],
                    style={'margin-bottom': '20px', 'align-items': 'center'},
                ),
                html.Div(
                    [
                        html.H4('Select playlist:'),
                        dcc.Dropdown(
                            id='user-song-preference-analysis-playlist-dropdown',
                            options=[{'label': 'default', 'value': 'default', 'disabled': True}],
                        ),
                    ],
                    style={'margin-bottom': '20px'},  # Add margin bottom
                ),
                dcc.Graph(
                    id='user-song-preference-analysis-figure',
                    className='mb-4'
                ),
                html.P('Note:'),
                html.Br(),
                html.P('Radar charts display different song features on multiple axes, with each axis representing a specific attribute like danceability or energy. The distance from the center on each axis indicates the preference or value for that feature.')
            ],
            style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
        )
    elif pathname == "/similar-user-matching":
        return dbc.Container(
            html.Div([
                html.H2('Similar User Matching'),
                html.H4(children="Select user "),
                dcc.Dropdown(
                    id="similar-user-matching-dropdown",
                    options=[{'label': user, 'value': user} for user in spotify_data['user_id'].unique()],
                    value=spotify_data['user_id'][0],
                ),
                html.Div([
                    html.H4(children="Match user according to: "),
                    dbc.RadioItems(
                        id="similar-user-matching-radio", 
                        options=[{'label': i, 'value': i} for i in ['Your specific music taste', 'Artists you have in common']], 
                        value='Your specific music taste',
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-secondary",
                        labelCheckedClassName="active",
                        style={'display' : 'flex'}
                    ),
                ], style={'margin-top': '10px', 'margin-bottom' : '10px'}),
                html.Div( id="similar-user-matching-collaborative-slider-div", children=[
                    html.H4(children="Adjust threshold (Min. number of artists in common)"),
                    dcc.Slider(id="similar-user-matching-collaborative-slider", min=1, step=1, value=1)
                ], style={'display': 'none'}),
                html.Div( id="similar-user-matching-content-based-slider-div", children=[
                    html.H4(children="Adjust threshold (Number of features with values equal or more than your values.)"),
                    dcc.Slider(id="similar-user-matching-content-based-slider", min=0, max=13, step=1, value=4),
                    # Just to get rid of initial missing id callback errors
                    # html.Div(sliders),
                ], style={'display': 'hidden'}),
                dcc.Graph(
                    id='similar-user-matching-figure1',
                ),
                dcc.Graph(
                    id='similar-user-matching-figure2',
                ),
                html.Div(id="similar-user-matching-table"),
                html.Br(),
                html.P('Note:'),
                html.Br(),
                html.P('The first graph with precision, recall, and f1-score evaluates the performance of our recommendation algorithm. It measures how accurately the algorithm identifies similar users based on their music preferences.'),
                html.Br(),
                html.P("The second graph is a line chart displaying evaluation metrics at different thresholds. This chart illustrates the algorithm's performance concerning varying thresholds set by users. It helps users understand how changing the threshold impacts the quality and relevance of recommendations. Adjusting the threshold allows users to fine-tune recommendations and controls the models strictness.")
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
        )
    elif pathname == "/song-clustering":
        return dbc.Container(
            html.Div([
                html.H2('Song Clustering & Exploration', style={'margin-bottom': '20px'}),
                html.Div([
                    html.H4(children="Select song features"),
                    dcc.Dropdown(
                        id='song-clustering-features-dropdown',
                        options=[
                            {'label': 'Danceability', 'value': 'danceability'},
                            {'label': 'Energy', 'value': 'energy'},
                            {'label': 'Loudness', 'value': 'loudness'},
                            {'label': 'Speechiness', 'value': 'speechiness'},
                            {'label': 'Acousticness', 'value': 'acousticness'},
                            {'label': 'Instrumentalness', 'value': 'instrumentalness'},
                            {'label': 'Liveness', 'value': 'liveness'},
                            {'label': 'Valence', 'value': 'valence'},
                            {'label': 'Tempo', 'value': 'tempo'},
                        ],
                        value=['danceability', 'energy'],
                        multi=True,
                        clearable=False,
                        style={'margin-bottom': '20px'}
                    ),
                ], style={'margin-bottom': '20px'}),
                html.Div(id="song-clustering-slider-div", children=[
                    html.H4(children="Select the number of clusters"),
                    dcc.Slider(id="song-clustering-slider", min=2, max=15, step=1, value=5),
                ], style={'margin-bottom': '20px'}),
                dcc.Graph(
                    id='song-clustering-figure',
                ),
                html.Br(),
                html.P('Note:'),
                html.Br(),
                html.P("In the scatter plot for clusters, each point represents a song. Songs with similar characteristics, like danceability and energy, are grouped together and assigned the same color or symbol, indicating their cluster. By observing the distribution of points, you can visually identify clusters or groups of songs that share common attributes. This visualization allows for the exploration of patterns and similarities among songs based on the selected features, aiding in understanding the inherent relationships between them."),
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            }),
        )
    elif pathname == "/genre-identification":
        return dbc.Container(
            html.Div([
                html.H2('Genre Identification'),
                
                html.H4(children="Set Learning Rate"),
                dcc.Slider(
                    id="genre-identification-learning-rate-slider",
                    min=0, max=1, value=0.001,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                
                html.H4(children="1. Model Evaluation - Accuracy & Loss Curve", style = {'margin-top' : '20px'}),
                dcc.Graph(
                    id='genre-identification-figure1',
                    style={'margin-bottom': '20px', 'margin-top': '10px'}
                ),
                
                html.H4(children="2. Model Evaluation - Confusion Matrix"),
                dcc.Graph(
                    id='genre-identification-figure2',
                    style={'margin-bottom': '20px', 'margin-top': '10px'}
                ),
                
                html.H4(children="Your Data Evaluation:"),
                html.Div([
                    html.H4('Select user:'),
                    dcc.Dropdown(
                        id="genre-identification-dropdown",
                        options=[{'label': user, 'value': user} for user in spotify_data['user_id'].unique()],
                        value=spotify_data['user_id'][0],
                    ),
                ], style={'margin-bottom': '20px'}),  # Add margin bottom
                
                html.Div([
                    dcc.Graph(
                        id='genre-identification-figure3',
                    )
                ], style={'margin-bottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H4(children="Your most frequently listened genre:", style={"margin-right": "20px"}),
                        html.H4(id='genre-identification-most-listened-genre')
                    ], style={'display': 'flex'}),
                    html.Div([
                        dcc.Graph(
                            id='genre-identification-figure4',
                            style={'margin-bottom': '20px', 'margin-top': '10px'}
                        ),
                    ])
                ], style={'display': 'flex', 'flex-direction': 'column', 'margin-bottom': '20px'}),
                
                html.P('Note:'),
                html.Br(),
                html.P('1. Model Evaluation - Accuracy & Loss Curve (line chart): This graph shows how well the machine learning model performs over time. The accuracy curve indicates how often the model correctly identifies genres, while the loss curve illustrates how the model learns and improves its predictions.'),
                html.Br(),
                html.P('2. Model Evaluation - Confusion Matrix: The confusion matrix visualizes the model`s performance by displaying the count of correct and incorrect genre predictions. It helps in understanding where the model makes mistakes and how well it classifies different genres.'),
                html.Br(),
                html.P('3. User Overall Genre Preference: This chart presents the distribution of your most frequently listened genres. It shows the number of times each genre appears, giving an overview of users musical preferences.'),
                html.Br(),
                html.P("4. User Aggregated Genre Preference: This chart offers a comprehensive view of user's music taste across multiple genres. Each axis represents a different genre, and the plot's shape indicates user's preferences for various music genres, allowing user's to see their overall music listening pattern.")
            ], style={
                'background-color': '#f2f2f2',  # Solid background color
                'border-radius': '10px',  # Slight round border
                'padding': '20px',  # Padding to add space between content and border
                'margin-top': '10px',
                'margin-bottom': '10px',
            })
        )

    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=True)