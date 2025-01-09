# %%
import os
import models.hdf5_getters as hdf5_getters
import pandas as pd

dataset_path = '/Users/karthik1705/Desktop/Projects/mu-seek/data'
df = pd.DataFrame()

# Looping through all .h5 files in the dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.h5'):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")

            # Open the HDF5 file
            h5 = hdf5_getters.open_h5_file_read(file_path)

            # Get song information
            duration = hdf5_getters.get_duration(h5)
            title = hdf5_getters.get_title(h5).decode()
            artist_name = hdf5_getters.get_artist_name(h5).decode()
            danceability = hdf5_getters.get_danceability(h5)
            num_songs = hdf5_getters.get_num_songs(h5)
            artist_hotttnesss = hdf5_getters.get_artist_hotttnesss(h5)
            energy = hdf5_getters.get_energy(h5)
            loudness = hdf5_getters.get_loudness(h5)
            tempo = hdf5_getters.get_tempo(h5)
            year = hdf5_getters.get_year(h5)

            df =pd.concat([df, pd.DataFrame([{'Artist': artist_name, 'Song': title, 'Duration': duration,
                                              'Danceability': danceability, 'Number of Songs': num_songs,
                                               'Artist Hotness': artist_hotttnesss, 'Energy': energy,
                                               'Loudness': loudness, 'Tempo': tempo, 'Year': year}])], ignore_index=True)
            # Close the file
            h5.close()

# Print song details
print(df)


# %%

# List of numerical columns
columns_to_plot = ['Duration', 'Artist Hotness', 'Energy', 'Loudness', 'Tempo']

fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(15, 5), sharey=True)

# Plotting histograms to observe numerical distributions
for ax, column in zip(axes, columns_to_plot):
    ax.hist(df[column], bins=30, alpha=0.7, edgecolor='black')
    ax.set_title(column)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# Checking for NULLs
print(df.isnull().sum())

# %%
# Splitting columns based on types
numeric_columns = ['Duration', 'Artist Hotness', 'Energy', 'Loudness', 'Tempo']
metadata_columns = ['Artist', 'Song']
data = df

# %%
# Scaling the numerical columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data[numeric_columns])
print(normalized_data)

# %%
# Implementing a Content-Based recommender algorithm - Cosine_Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity matrix
similarity_matrix = cosine_similarity(normalized_data)

# Finding similar songs to the first song
similar_songs = similarity_matrix[0]
print(similar_songs)

# %%
# Fetching the top 5 similar songs to the first song
import numpy as np

# Get indices of top 5 most similar songs
top_indices = np.argsort(similar_songs)[::-1][1:6]  # Excluding the song itself
print(f"Recommended Songs: {top_indices}")

# %%
# Fetching the artist and song names
metadata = df[metadata_columns]
for idx in top_indices:
    artist = metadata.iloc[idx]['Artist']
    song = metadata.iloc[idx]['Song']
    print(f"Song: {song}, Artist: {artist}")

# %%
# Latent Feature Extraction (SVD)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5, random_state=42)
latent_features = svd.fit_transform(normalized_data)
print(latent_features.shape)

# %%
# Cosine Similarity using latent feature space

# Compute the cosine similarity matrix
similarity_matrix_svd = cosine_similarity(latent_features)

# Getting the similarity scores for the first song again
similar_songs_svd = similarity_matrix_svd[0]

top_indices_svd = np.argsort(similar_songs_svd)[::-1][1:6]  # Top 5 similar songs (excluding itself)
print(f"Top 5 similar songs: {top_indices_svd}")


# %%
# Fetching the artist and song names
metadata = df[metadata_columns]
for idx in top_indices_svd:
    artist = metadata.iloc[idx]['Artist']
    song = metadata.iloc[idx]['Song']
    print(f"Song: {song}, Artist: {artist}")

# %% Implementing Latent Feature Extraction using Euclidean distance
from sklearn.metrics.pairwise import euclidean_distances

# Compute the Euclidean distance matrix
distance_matrix_svd = euclidean_distances(latent_features)

# Getting the most similar songs (smallest distances)
similar_songs_euc_svd = distance_matrix_svd[0]

# Sort songs by distance (smaller values mean more similar)
top_indices_euc_svd = np.argsort(similar_songs_euc_svd)[1:6]  # Top 5 closest songs
print(f"Top 5 similar songs: {top_indices_euc_svd}")

# %%
# Fetching the artist and song names
metadata = df[metadata_columns]
for idx in top_indices_euc_svd:
    artist = metadata.iloc[idx]['Artist']
    song = metadata.iloc[idx]['Song']
    print(f"Song: {song}, Artist: {artist}")
