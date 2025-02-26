import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
from scipy import sparse
from typing import Union
import sys

# Configure logging to show more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Set page config at the very beginning
st.set_page_config(
    page_title="NutriPal Recipe Recommender",
    page_icon="🍳",
    layout="wide"
)

# ---------------------
# CACHED FUNCTIONS
# ---------------------
@st.cache_resource
def load_model(model_path: str):
    """Load a pre-trained model from a pickle file."""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load the recommendation model. Please check the model file.")
        return None

@st.cache_resource
def load_vectorizer(vectorizer_path: str):
    """Load a pre-trained TF-IDF vectorizer from a pickle file."""
    try:
        vectorizer_path = Path(vectorizer_path)
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading vectorizer: {str(e)}")
        st.error("Failed to load the text vectorizer. Please check the vectorizer file.")
        return None

@st.cache_data
def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded columns from {file_path}: {df.columns.tolist()}")
        
        # Check if required columns exist
        if 'ingredients' not in df.columns or 'steps' not in df.columns:
            raise ValueError(f"Missing required columns. Available columns: {df.columns.tolist()}")
        
        # Check if calories column exists or needs to be calculated/renamed
        if 'calories' not in df.columns:
            # Check for alternative column names
            calorie_columns = [col for col in df.columns if 'calor' in col.lower()]
            if calorie_columns:
                logger.info(f"Using {calorie_columns[0]} as calories column")
                df['calories'] = df[calorie_columns[0]]
            else:
                logger.warning("No calories column found. Setting default value.")
                df['calories'] = 500  # Default value
        
        # Clean and preprocess data
        df['ingredients'] = df['ingredients'].astype(str).fillna('')
        df['steps'] = df['steps'].astype(str).fillna('')
        df['text'] = df['ingredients'] + ' ' + df['steps']
        
        # Handle missing values in calories
        df['calories'] = pd.to_numeric(df['calories'], errors='coerce')
        df = df.dropna(subset=['calories'])
        
        # Ensure 'name' column exists
        if 'name' not in df.columns and 'recipe_name' in df.columns:
            df['name'] = df['recipe_name']
        elif 'name' not in df.columns:
            df['name'] = [f"Recipe {i+1}" for i in range(len(df))]
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load the recipe data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_features_cached(text_data: list, _vectorizer) -> Union[sparse.csr_matrix, None]:
    """Transform the text data into TF-IDF features."""
    try:
        return _vectorizer.transform(text_data)
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return None

def get_features(_vectorizer, df: pd.DataFrame):
    """Wrapper function to handle the vectorizer transformation."""
    if _vectorizer is None or df.empty:
        return None
    return get_features_cached(df['text'].tolist(), _vectorizer)

# ---------------------
# RECOMMENDATION FUNCTION
# ---------------------
def recommend_recipes(
    user_preferences: str,
    calories_limit: int,
    cooking_time_option: str,
    data: pd.DataFrame,
    features,
    content_vectorizer,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Recommend recipes based on user preferences, calorie limit, and cooking time.
    """
    try:
        if data.empty or features is None:
            return pd.DataFrame()
        
        # Filter recipes based on calorie limit
        mask = data['calories'] <= calories_limit
        
        # Additional time filtering if 'minutes' column exists
        if 'minutes' in data.columns:
            if cooking_time_option == "Less than 30 minutes":
                mask &= data['minutes'] < 30
            elif cooking_time_option == "Between 30 to 60 minutes":
                mask &= (data['minutes'] >= 30) & (data['minutes'] <= 60)
            else:  # More than 60 minutes
                mask &= data['minutes'] > 60
        
        filtered_data = data[mask]
        
        if filtered_data.empty:
            return pd.DataFrame()
        
        # Transform user preferences
        user_pref_features = get_features_cached([user_preferences], content_vectorizer)
        
        # Calculate similarities
        filtered_features = get_features_cached(filtered_data['text'].tolist(), content_vectorizer)
        similarity_scores = cosine_similarity(user_pref_features, filtered_features)
        
        # Get recommendations
        top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        columns_to_show = ['name', 'ingredients', 'steps', 'calories']
        if 'minutes' in filtered_data.columns:
            columns_to_show.append('minutes')
            
        recommendations = filtered_data.iloc[top_indices][columns_to_show].copy()
        
        # Format the output
        recommendations['calories'] = recommendations['calories'].round(1)
        if 'minutes' in recommendations.columns:
            recommendations['minutes'] = recommendations['minutes'].round(0)
        
        return recommendations.reset_index(drop=True)
    
    except Exception as e:
        logger.error(f"Error in recipe recommendation: {str(e)}")
        st.error("An error occurred while generating recommendations.")
        return pd.DataFrame()

# ---------------------
# STREAMLIT APP
# ---------------------
def main():
    st.title("Recipe Intelligence Recommender")
    st.subheader("WHAT IS BEST DISH FOR YOU!!!")
    
    # Load dependencies
    with st.spinner("Loading recommendation system..."):
        try:
            model = load_model('content_user_model.pkl')
            content_vectorizer = load_vectorizer('tfidf_vectorizer.pkl')
            African_data = load_and_process_data('cleaned_African_recipies.csv')
            International_data = load_and_process_data('cleaned_International_recipies.csv')
            
            # Log data loading results
            logger.info(f"African data shape: {African_data.shape}")
            logger.info(f"International data shape: {International_data.shape}")
            
            # Generate features for both datasets
            if content_vectorizer is not None:
                African_features = get_features(content_vectorizer, African_data)
                International_features = get_features(content_vectorizer, International_data)
                
                if African_features is None or International_features is None:
                    st.error("Failed to generate features for the recipes.")
                    return
            else:
                st.error("Failed to initialize the recommendation system. Please check the vectorizer file.")
                return
                
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            st.error("Failed to initialize the recommendation system. Please check the log for details.")
            return
    
    # User input section
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            user_input = st.text_area(
                "Enter your preferences",
                placeholder="e.g., 'I want a high-protein meal with chicken'",
                help="Describe what kind of recipe you're looking for"
            )
            
            recipe_type = st.selectbox(
                "Select Recipe Type",
                ["International", "African"],
                help="Choose the type of cuisine you prefer"
            )
        
        with col2:
            calories_limit = st.number_input(
                "Maximum calories",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
                help="Set the maximum calories for your meal"
            )
            
            cooking_time_option = st.selectbox(
                "Cooking Time",
                ["Less than 30 minutes", "Between 30 to 60 minutes", "More than 60 minutes"],
                help="Select your preferred cooking time range"
            )
    
    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        if not user_input.strip():
            st.warning("Please enter your recipe preferences.")
            return
            
        with st.spinner("Finding the perfect recipes for you..."):
            if recipe_type == "African":
                recommendations = recommend_recipes(
                    user_input, calories_limit, cooking_time_option,
                    African_data, African_features, content_vectorizer
                )
            else:
                recommendations = recommend_recipes(
                    user_input, calories_limit, cooking_time_option,
                    International_data, International_features, content_vectorizer
                )
            
            if recommendations.empty:
                st.warning("No recipes found matching your criteria. Try adjusting your preferences.")
            else:
                st.success("Here are your personalized recipe recommendations!")
                
                # Display recommendations in an expandable format
                for idx, row in recommendations.iterrows():
                    with st.expander(f"📝 {row['name']}"):
                        st.write("**Calories:** ", row['calories'])
                        if 'minutes' in row:
                            st.write("**Cooking Time:** ", f"{int(row['minutes'])} minutes")
                        
                        st.write("**Ingredients:**")
                        st.write(row['ingredients'])
                        
                        st.write("**Steps:**")
                        st.write(row['steps'])

if __name__ == "__main__":
    main()