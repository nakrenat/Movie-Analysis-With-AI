import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def load_imdb_data():
    """
    Loads IMDb data from the Kaggle dataset.
    """
    try:
        # Read the dataset downloaded from Kaggle
        df = pd.read_csv('IMDB-Movie-Data.csv')
        return df
    except FileNotFoundError:
        print("Dataset not found. Please add 'IMDB-Movie-Data.csv' file to the project directory.")
        return None

def analyze_genre_ratings(df):
    """
    Analyzes and visualizes average ratings by movie genre.
    """
    # Split genres and create separate rows for each genre
    genre_df = df.copy()
    genre_df['Genre'] = genre_df['Genre'].str.split(',')
    genre_df = genre_df.explode('Genre')
    
    # Calculate average ratings by genre
    genre_ratings = genre_df.groupby('Genre')['Rating'].agg(['mean', 'count']).reset_index()
    genre_ratings = genre_ratings[genre_ratings['count'] >= 10]  # Only include genres with at least 10 movies
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(data=genre_ratings, x='Genre', y='mean')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average IMDb Ratings by Movie Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')
    plt.tight_layout()
    plt.savefig('genre_ratings.png')
    plt.close()

def analyze_year_ratings(df):
    """
    Analyzes and visualizes average ratings by year.
    """
    # Calculate average ratings by year
    year_ratings = df.groupby('Year')['Rating'].mean().reset_index()
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=year_ratings, x='Year', y='Rating')
    plt.title('Average IMDb Ratings by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('year_ratings.png')
    plt.close()

def analyze_correlation(df):
    """
    Analyzes correlation between numerical variables.
    """
    # Calculate correlation matrix
    numeric_cols = ['Rating', 'Votes', 'Revenue (Millions)', 'Metascore', 'Runtime (Minutes)']
    corr_matrix = df[numeric_cols].corr()
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Between Variables')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def perform_regression_analysis(df):
    """
    Performs simple linear regression analysis.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df['Metascore'].values.reshape(-1, 1))
    y = df['Rating'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    plt.title('Metascore vs IMDb Rating')
    plt.xlabel('Metascore')
    plt.ylabel('IMDb Rating')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('regression_analysis.png')
    plt.close()
    
    print(f"R-squared value: {model.score(X, y):.3f}")

def analyze_language_country(df):
    """
    Analyzes and visualizes ratings by language and country.
    """
    # Language analysis
    df['Language'] = df['Language'].str.split(',')
    df = df.explode('Language')
    
    # Get top 10 languages by number of movies
    top_languages = df['Language'].value_counts().head(10).index
    language_ratings = df[df['Language'].isin(top_languages)].groupby('Language')['Rating'].mean().sort_values(ascending=False)
    
    # Visualization for languages
    plt.figure(figsize=(12, 6))
    language_ratings.plot(kind='bar')
    plt.title('Average IMDb Ratings by Language (Top 10)')
    plt.xlabel('Language')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('language_ratings.png')
    plt.close()
    
    # Country analysis
    df['Country'] = df['Country'].str.split(',')
    df = df.explode('Country')
    
    # Get top 10 countries by number of movies
    top_countries = df['Country'].value_counts().head(10).index
    country_ratings = df[df['Country'].isin(top_countries)].groupby('Country')['Rating'].mean().sort_values(ascending=False)
    
    # Visualization for countries
    plt.figure(figsize=(12, 6))
    country_ratings.plot(kind='bar')
    plt.title('Average IMDb Ratings by Country (Top 10)')
    plt.xlabel('Country')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('country_ratings.png')
    plt.close()

def analyze_revenue(df):
    """
    Analyzes and visualizes movie revenue patterns.
    """
    # Create a scatter plot of revenue vs rating
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Rating'], df['Revenue (Millions)'], alpha=0.5)
    plt.title('Movie Rating vs Revenue')
    plt.xlabel('IMDb Rating')
    plt.ylabel('Revenue (Millions $)')
    plt.grid(True)
    
    # Add trend line
    mask = ~np.isnan(df['Revenue (Millions)'])
    z = np.polyfit(df['Rating'][mask], df['Revenue (Millions)'][mask], 1)
    p = np.poly1d(z)
    plt.plot(df['Rating'][mask], p(df['Rating'][mask]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('rating_revenue.png')
    plt.close()
    
    # Get top 10 movies by revenue
    top_revenue = df.nlargest(10, 'Revenue (Millions)')[['Title', 'Rating', 'Revenue (Millions)', 'Year']]
    print("\nTop 10 Movies by Revenue:")
    print(top_revenue.to_string(index=False))
    
    # Calculate correlation
    correlation = df['Rating'].corr(df['Revenue (Millions)'])
    print(f"\nCorrelation between Rating and Revenue: {correlation:.3f}")

def analyze_runtime_rating(df):
    """
    Analyzes and visualizes the relationship between movie runtime and rating.
    """
    # Create a scatter plot of runtime vs rating
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Runtime (Minutes)'], df['Rating'], alpha=0.5)
    plt.title('Movie Runtime vs Rating')
    plt.xlabel('Runtime (Minutes)')
    plt.ylabel('IMDb Rating')
    plt.grid(True)
    
    # Add trend line
    mask = ~np.isnan(df['Runtime (Minutes)'])
    z = np.polyfit(df['Runtime (Minutes)'][mask], df['Rating'][mask], 1)
    p = np.poly1d(z)
    plt.plot(df['Runtime (Minutes)'][mask], p(df['Runtime (Minutes)'][mask]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('runtime_rating.png')
    plt.close()
    
    # Calculate correlation
    correlation = df['Runtime (Minutes)'].corr(df['Rating'])
    print(f"\nCorrelation between Runtime and Rating: {correlation:.3f}")

def train_prediction_model(df):
    """
    Trains a machine learning model to predict movie ratings.
    """
    print("\nPreparing features for machine learning...")
    # Prepare features
    df = prepare_features(df)
    
    # Select features for the model
    feature_cols = [col for col in df.columns if col.startswith('Genre_')] + [
        'Director_Encoded', 'Runtime (Minutes)', 'Votes', 'Revenue (Millions)',
        'Metascore', 'Year'
    ]
    
    X = df[feature_cols]
    y = df['Rating']
    
    print(f"Total number of features used: {len(feature_cols)}")
    print("Features used:", feature_cols)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"R-squared Score: {r2:.3f}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head().to_string())
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features for Rating Prediction')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Movie Ratings')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_performance.png')
    plt.close()
    
    # Example predictions
    print("\nExample Predictions (Actual vs Predicted):")
    example_df = pd.DataFrame({
        'Actual': y_test[:5],
        'Predicted': y_pred[:5]
    })
    print(example_df.to_string())
    
    return model, feature_importance

def prepare_features(df):
    """
    Prepares features for machine learning model.
    """
    # Create a copy of the dataframe
    df = df.copy()
    
    # Create genre features (one-hot encoding)
    df['Genre'] = df['Genre'].str.split(',')
    genres = df['Genre'].explode().unique()
    for genre in genres:
        df[f'Genre_{genre}'] = df['Genre'].apply(lambda x: 1 if genre in x else 0)
    
    # Create director features
    le = LabelEncoder()
    df['Director_Encoded'] = le.fit_transform(df['Director'])
    
    # Handle missing values in numerical features
    imputer = SimpleImputer(strategy='mean')
    numerical_features = ['Runtime (Minutes)', 'Votes', 'Revenue (Millions)', 'Metascore']
    df[numerical_features] = imputer.fit_transform(df[numerical_features])
    
    return df

def main():
    # Load data
    df = load_imdb_data()
    if df is None:
        return
    
    # Perform analyses
    print("Analyzing ratings by genre...")
    analyze_genre_ratings(df)
    
    print("Analyzing ratings by year...")
    analyze_year_ratings(df)
    
    print("Performing correlation analysis...")
    analyze_correlation(df)
    
    print("Performing regression analysis...")
    perform_regression_analysis(df)
    
    print("Analyzing revenue patterns...")
    analyze_revenue(df)
    
    print("Analyzing runtime and rating relationship...")
    analyze_runtime_rating(df)
    
    print("\nTraining machine learning model...")
    model, feature_importance = train_prediction_model(df)
    
    print("\nAnalysis complete! Results have been visualized.")

if __name__ == "__main__":
    main() 