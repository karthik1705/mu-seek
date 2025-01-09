# Mu-Seek: A Personalized Music Recommender System 🎵

Mu-Seek is a personalized music recommendation system that suggests songs based on user preferences and song features. It leverages advanced machine learning techniques like collaborative filtering, content-based filtering, and hybrid models to deliver tailored recommendations.

## Features
- 🎧 **Content-Based Recommendations**: Suggests songs similar to the ones a user likes based on audio features like tempo, energy, and danceability.
- 🤝 **Collaborative Filtering**: Recommends songs based on the preferences of similar users.
- 🛠️ **Hybrid Approach**: Combines content-based and collaborative methods for improved accuracy.
- 🌟 **Real-Time Feedback**: Learns from user interactions (likes, skips, etc.) to improve recommendations.

## Dataset
- Uses the [Million Song Dataset](http://millionsongdataset.com/) for the current prototype.
- Plan is to integrate with the Spotify API for dynamic recommendations.
- Audio features are extracted using tools like **Librosa**.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/karthik1705/mu-seek.git
   cd mu-seek
