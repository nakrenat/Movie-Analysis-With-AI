# IMDb Movie Rating Analysis

This Python application performs various analyses on IMDb movie data and includes a machine learning model for rating prediction.

## Features

- Average rating analysis by movie genre
- Movie rating analysis by year
- Correlation analysis between variables
- Simple linear regression analysis
- Revenue analysis and patterns
- Runtime vs rating relationship analysis
- Machine learning model for rating prediction
  - Random Forest Regressor
  - Feature importance analysis
  - Model performance evaluation

## Setup

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Download the IMDb dataset from Kaggle:
   - Download from [IMDB Movie Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data)
   - Copy the downloaded `IMDB-Movie-Data.csv` file to the project directory

## Usage

To run the analysis:

```bash
python movie_analysis.py
```

The program will generate the following visualizations:
- `genre_ratings.png`: Average ratings by movie genre
- `year_ratings.png`: Average ratings by year
- `correlation_matrix.png`: Correlation matrix between variables
- `regression_analysis.png`: Relationship between Metascore and IMDb Rating
- `rating_revenue.png`: Relationship between movie ratings and revenue
- `runtime_rating.png`: Relationship between movie runtime and rating
- `feature_importance.png`: Most important features for rating prediction
- `prediction_performance.png`: Actual vs predicted ratings comparison

## Analysis Results

When the program runs, it will create visualizations and print summary information to the console. These visualizations will show:

- Which movie genres receive higher ratings
- How movie ratings have changed over the years
- Relationships between different variables
- Correlation between Metascore and IMDb Rating
- How movie ratings affect revenue
- How movie runtime affects ratings
- Which features are most important for predicting movie ratings
- How well the machine learning model performs

## Machine Learning Model

The application includes a Random Forest Regressor model that:
- Predicts movie ratings based on various features
- Uses genre, director, runtime, votes, revenue, and other features
- Provides feature importance analysis
- Shows model performance metrics (MSE and R-squared)
- Visualizes actual vs predicted ratings 