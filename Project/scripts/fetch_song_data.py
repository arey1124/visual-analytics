import requests
import pandas as pd

file_path = 'C:/Users/ariha/OneDrive/Desktop/DAL MACS/CSCI6612-VisualAnalytics/Project/spotify_dataset.csv'

df = pd.read_csv(file_path, nrows=25000)
# Remove extra unnamed columns with NaN values
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Check for rows with all blank values (only commas or empty cells)
blank_rows = df.apply(lambda row: all(pd.isna(row) | row.astype(str).str.strip().eq('')), axis=1)

# Filter and keep rows that are not entirely blank
df = df[~blank_rows]

# Replace with your Spotify Developer App credentials
CLIENT_ID = 'c1104735ff754f3c91cbad00fd10e5c0'
CLIENT_SECRET = '320c5946d21445e28b3ae0d7dc13575b'

# Authenticate and obtain an access token
auth_url = 'https://accounts.spotify.com/api/token'
auth_data = {'grant_type': 'client_credentials'}
auth_response = requests.post(auth_url, auth=(CLIENT_ID, CLIENT_SECRET), data=auth_data)
auth_response_data = auth_response.json()
access_token = auth_response_data['access_token']

# Set up the API request headers
headers = {
    'Authorization': f'Bearer {access_token}'
}

def fetch_song_details(artist, track_name):
    # Define the search query
    query = f'artist:{artist} track:{track_name}'

    # Make the API request to search for the song
    search_url = f'https://api.spotify.com/v1/search?q={query}&type=track'
    response = requests.get(search_url, headers=headers)
    search_results = response.json()
    # Extract the first track's ID (URI)
    if 'tracks' in search_results and 'items' in search_results['tracks'] and search_results['tracks']['items']:
        track_id = search_results['tracks']['items'][0]['uri']
        print(f"Spotify Song ID (URI): {track_id}, artist : {artist}")

        return track_id.split(":")[-1]
    else:
        print(f"Song not found on Spotify, artist : {artist}")

def fetch_song_features(track_id):
    # Fetch audio features for the track using its URI
    audio_features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'  # Extract track ID from URI
    audio_features_response = requests.get(audio_features_url, headers=headers)
    return audio_features_response.json()

# Function to populate data from API response into DataFrame
def populate_data(row):
    if not row['song_id']:
        return row
    api_response = fetch_song_features(row['song_id'])
    
    # Check if each field exists in api_response before assigning it
    if 'danceability' in api_response:
        row['danceability']  = api_response['danceability']
    if 'energy' in api_response:
        row['energy'] = api_response['energy']
    if 'key' in api_response:
        row['key'] = api_response['key']
    if 'loudness' in api_response:
        row['loudness'] = api_response['loudness']
    if 'mode' in api_response:
        row['mode'] = api_response['mode']
    if 'speechiness' in api_response:
        row['speechiness'] = api_response['speechiness']
    if 'acousticness' in api_response:
        row['acousticness'] = api_response['acousticness']
    if 'instrumentalness' in api_response:
        row['instrumentalness'] = api_response['instrumentalness']
    if 'liveness' in api_response:
        row['liveness'] = api_response['liveness']
    if 'valence' in api_response:
        row['valence'] = api_response['valence']
    if 'tempo' in api_response:
        row['tempo'] = api_response['tempo']
    if 'type' in api_response:
        row['type'] = api_response['type']
    if 'duration_ms' in api_response:
        row['duration_ms'] = api_response['duration_ms']
    if 'time_signature' in api_response:
        row['time_signature'] = api_response['time_signature']
    if 'analysis_url' in api_response:
        row['analysis_url'] = api_response['analysis_url']
    if 'uri' in api_response:
        row['uri'] = api_response['uri']
    if 'id' in api_response:
        row['id'] = api_response['id']
    return row

# Create a new column 'song_id' in the DataFrame
df['song_id'] = df.apply(lambda row: fetch_song_details(row[' artistname'], row[' trackname']), axis=1)
# Use apply to populate data for each row
#df = df.apply(lambda row: populate_data(row), axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_dataset.csv', mode='a', index=False)
    