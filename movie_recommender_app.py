import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import zlib

class MovieRecommender:
    def __init__(self, data_dir="amazon_movies_2023"):
        # Load embeddings
        embeddings_path = os.path.join(data_dir, "title_embeddings.npz")
        try:
            embeddings_data = np.load(embeddings_path)
            self.embeddings = embeddings_data['embeddings']
            self.item_ids = embeddings_data['item_ids']
        except (IOError, zlib.error) as e:
            raise RuntimeError(
                f"Error loading embeddings file: {str(e)}\n"
                "The embeddings file appears to be corrupted or invalid.\n"
                "Please regenerate the embeddings by running create_title_embeddings.py first."
            )
        
        # Load movie mapping
        mapping_path = os.path.join(data_dir, "title_embeddings_mapping.csv")
        self.movies_df = pd.read_csv(mapping_path)
        
        # Standardize embeddings
        scaler = StandardScaler()
        self.embeddings = scaler.fit_transform(self.embeddings)
        
        # Create item_id to index mapping for faster lookups
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        # Create title to id mapping for search
        self.title_to_id = dict(zip(self.movies_df['title'], self.movies_df['item_id']))
        
        # Store all titles for search
        self.all_titles = self.movies_df['title'].tolist()
        
    def search_movies(self, query):
        if not query:
            return []
        # Case-insensitive search
        query = query.lower()
        matches = [
            title for title in self.all_titles
            if query in title.lower()
        ]
        return matches[:10]  # Return top 10 matches
        
    def get_recommendations(self, selected_titles, n_recommendations=10):
        if not selected_titles:
            return []
            
        # Filter out any invalid titles
        selected_titles = [title for title in selected_titles if title in self.title_to_id]
        
        if not selected_titles:
            return []
            
        # Get indices of selected movies
        selected_indices = [
            self.id_to_idx[self.title_to_id[title]]
            for title in selected_titles
        ]
        
        # Get embeddings of selected movies
        selected_embeddings = self.embeddings[selected_indices]
        
        # Calculate user embedding (average of selected movies)
        user_embedding = np.mean(selected_embeddings, axis=0)
        
        # Normalize user embedding for cosine similarity
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        
        # Calculate cosine similarity with all movies
        # Since embeddings are already normalized, dot product gives cosine similarity
        similarities = np.dot(self.embeddings, user_embedding)
        
        # Get indices of most similar movies (excluding selected ones)
        recommendation_indices = []
        # Sort by similarity (highest to lowest)
        idx_sorted = np.argsort(-similarities)  # Negative to sort in descending order
        
        for idx in idx_sorted:
            if idx not in selected_indices:
                recommendation_indices.append(idx)
                if len(recommendation_indices) == n_recommendations:
                    break
        
        # Get recommended movie titles
        recommended_ids = self.item_ids[recommendation_indices]
        recommended_movies = self.movies_df[self.movies_df['item_id'].isin(recommended_ids)]
        
        # Sort recommendations by similarity score
        similarity_scores = similarities[recommendation_indices]
        recommendations_with_scores = list(zip(recommended_movies['title'].tolist(), similarity_scores))
        recommendations_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return recommendations with scores
        return [(title, float(score)) for title, score in recommendations_with_scores]

def create_interface():
    try:
        recommender = MovieRecommender()
    except Exception as e:
        print(f"Error initializing recommender: {str(e)}")
        return None
    
    with gr.Blocks() as iface:
        gr.Markdown(
        """
        # Movie Recommender
        Select movies you've enjoyed, and get personalized recommendations based on your taste!
        
        **How to use:**
        1. Type to search for a movie
        2. Select it from the dropdown
        3. Add up to 5 movies
        4. Click 'Get Recommendations' to see similar movies
        
        The similarity score (0-1) shows how close each recommendation is to your selected movies.
        """
        )
        
        selected_movies = gr.State([])
        
        with gr.Row():
            with gr.Column():
                # Movie search and selection
                movie_search = gr.Dropdown(
                    choices=[],
                    label="Search and select a movie",
                    interactive=True,
                    allow_custom_value=True
                )
                
                # Display selected movies
                selected_display = gr.Textbox(
                    label="Your Selected Movies",
                    value="No movies selected yet",
                    interactive=False
                )
                
                # Clear selection button
                clear_btn = gr.Button("Clear Selection")
                
                # Get recommendations button
                recommend_btn = gr.Button("Get Recommendations", variant="primary")
            
            with gr.Column():
                # Display recommendations
                recommendations = gr.Textbox(
                    label="Recommended Movies",
                    value="Recommendations will appear here",
                    interactive=False,
                    lines=10
                )
        
        def update_search_options(query):
            if not query:
                return gr.Dropdown(choices=[])
            matches = recommender.search_movies(query)
            return gr.Dropdown(choices=matches)
        
        def add_movie(movie, current_movies):
            if not movie:
                return current_movies, format_selected_movies(current_movies)
                
            current_movies = current_movies or []
            if len(current_movies) >= 5:
                return current_movies, format_selected_movies(current_movies)
                
            if movie not in current_movies:
                current_movies.append(movie)
            
            return current_movies, format_selected_movies(current_movies)
        
        def clear_selection(current_movies):
            return [], "No movies selected yet"
        
        def format_selected_movies(movies):
            if not movies:
                return "No movies selected yet"
            return "\n".join(f"{i+1}. {movie}" for i, movie in enumerate(movies))
        
        def format_recommendations(recommendations):
            if not recommendations:
                return "No recommendations available yet"
            return "\n".join(
                f"{i+1}. {title} (similarity: {score:.3f})" 
                for i, (title, score) in enumerate(recommendations)
            )
        
        # Event handlers
        movie_search.change(
            fn=update_search_options,
            inputs=movie_search,
            outputs=movie_search
        )
        
        movie_search.select(
            fn=add_movie,
            inputs=[movie_search, selected_movies],
            outputs=[selected_movies, selected_display]
        )
        
        clear_btn.click(
            fn=clear_selection,
            inputs=[selected_movies],
            outputs=[selected_movies, selected_display]
        )
        
        recommend_btn.click(
            fn=lambda x: format_recommendations(recommender.get_recommendations(x)),
            inputs=[selected_movies],
            outputs=recommendations
        )
    
    return iface

if __name__ == "__main__":
    iface = create_interface()
    if iface is not None:
        iface.launch()
    else:
        print("\nPlease fix the issues above and try again.") 