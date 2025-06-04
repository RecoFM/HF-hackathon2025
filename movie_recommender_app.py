import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import zlib
from typing import Dict, List, Tuple, Optional, Literal
from mistralai import Mistral
import os
from dotenv import load_dotenv
from ranking_agent import rank_with_ai

load_dotenv()

class MovieRecommender:
    def __init__(self, data_dir: str = "amazon_movies_2023"):
        self.data_dir = data_dir
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        # Load both types of embeddings
        self.load_embeddings()
        
    def load_embeddings(self) -> None:
        # Load LLM embeddings
        llm_embeddings_path = os.path.join(self.data_dir, "title_embeddings.npz")
        try:
            llm_data = np.load(llm_embeddings_path)
            self.llm_embeddings = llm_data['embeddings']
            self.llm_item_ids = llm_data['item_ids'].astype(str)  # Ensure string type
            print(f"Loaded LLM embeddings with shape: {self.llm_embeddings.shape}")
            print(f"Number of LLM item IDs: {len(self.llm_item_ids)}")
        except (IOError, zlib.error) as e:
            raise RuntimeError(
                f"Error loading LLM embeddings file: {str(e)}\n"
                "The embeddings file appears to be corrupted or invalid."
            )
            
        # Load GCL embeddings
        gcl_embeddings_path = os.path.join(self.data_dir, "gcl_embeddings.npz")
        try:
            gcl_data = np.load(gcl_embeddings_path)
            self.gcl_embeddings = gcl_data['embeddings']
            self.gcl_item_ids = gcl_data['item_ids'].astype(str)  # Ensure string type
            print(f"Loaded GCL embeddings with shape: {self.gcl_embeddings.shape}")
            print(f"Number of GCL item IDs: {len(self.gcl_item_ids)}")
        except (IOError, zlib.error) as e:
            raise RuntimeError(
                f"Error loading GCL embeddings file: {str(e)}\n"
                "Please run gcl_embeddings.py first to generate GCL embeddings."
            )
        
        # Load movie mapping
        mapping_path = os.path.join(self.data_dir, "title_embeddings_mapping.csv")
        self.movies_df = pd.read_csv(mapping_path)
        self.movies_df['item_id'] = self.movies_df['item_id'].astype(str)  # Ensure string type
        
        # Create standardized embeddings for both types
        scaler = StandardScaler()
        self.llm_embeddings = scaler.fit_transform(self.llm_embeddings)
        self.gcl_embeddings = scaler.fit_transform(self.gcl_embeddings)
        
        # Create item_id to index mappings for both types
        self.llm_id_to_idx = {str(item_id): idx for idx, item_id in enumerate(self.llm_item_ids)}
        self.gcl_id_to_idx = {str(item_id): idx for idx, item_id in enumerate(self.gcl_item_ids)}
        
        # Create title to id mapping for search
        self.title_to_id = dict(zip(self.movies_df['title'], self.movies_df['item_id']))
        
        # Store all titles for search
        self.all_titles = self.movies_df['title'].tolist()
        
        print(f"Number of movies in mapping: {len(self.movies_df)}")
        print(f"Number of titles with LLM embeddings: {len(set(self.llm_id_to_idx.keys()) & set(self.title_to_id.values()))}")
        print(f"Number of titles with GCL embeddings: {len(set(self.gcl_id_to_idx.keys()) & set(self.title_to_id.values()))}")
        
    def search_movies(self, query: str) -> List[str]:
        if not query:
            return []
        # Case-insensitive search
        query = query.lower()
        matches = [
            title for title in self.all_titles
            if query in title.lower()
        ]
        return matches[:20]  # Return top 20 matches
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Mistral API"""
        try:
            response = self.mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=[text]  # Note: inputs should be a list
            )
            # Convert embedding to numpy array
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"Error getting embedding from Mistral API: {str(e)}")
            return None
        
    def get_recommendations(
        self, 
        selected_titles: List[str], 
        embedding_type: Literal["LLM", "LLM + GCL"] = "LLM",
        user_preferences: str = "",
        alpha: float = 0.5,
        n_recommendations: int = 20
    ) -> List[Tuple[str, float]]:
        if not selected_titles and not user_preferences:
            return []
            
        # Filter out any invalid titles
        selected_titles = [title for title in selected_titles if title in self.title_to_id]
        
        # Choose embeddings based on type
        embeddings = self.gcl_embeddings if embedding_type == "LLM + GCL" else self.llm_embeddings
        id_to_idx = self.gcl_id_to_idx if embedding_type == "LLM + GCL" else self.llm_id_to_idx
        
        # Initialize user embedding components
        history_embedding = None
        preference_embedding = None
            
        # Get history-based embedding if we have selected titles
        if selected_titles:
            # Get indices of selected movies
            selected_indices = []
            missing_titles = []
            for title in selected_titles:
                item_id = str(self.title_to_id[title])
                if item_id in id_to_idx:
                    selected_indices.append(id_to_idx[item_id])
                else:
                    missing_titles.append(title)
                    print(f"Warning: Movie '{title}' (ID: {item_id}) not found in {embedding_type} embeddings")
            
            if missing_titles:
                print(f"Movies not found in {embedding_type} embeddings: {', '.join(missing_titles)}")
            
            if selected_indices:
                # Get embeddings of selected movies
                selected_embeddings = embeddings[selected_indices]
                # Calculate history embedding (average of selected movies)
                history_embedding = np.mean(selected_embeddings, axis=0)
                # Normalize history embedding
                history_embedding = history_embedding / np.linalg.norm(history_embedding)
        
        # Get preference-based embedding if we have user preferences
        if user_preferences:
            preference_embedding = self.get_text_embedding(user_preferences)
            if preference_embedding is None:
                print("Warning: Failed to get embedding for user preferences")
                if history_embedding is None:
                    return []
        
        # Combine embeddings based on availability and alpha
        if history_embedding is not None and preference_embedding is not None:
            # Both available - use alpha for weighted combination
            user_embedding = alpha * preference_embedding + (1 - alpha) * history_embedding
        elif history_embedding is not None:
            # Only history available
            user_embedding = history_embedding
        elif preference_embedding is not None:
            # Only preferences available
            user_embedding = preference_embedding
        else:
            return []
        
        # Normalize final user embedding
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        
        # Calculate cosine similarity with all movies
        similarities = np.dot(embeddings, user_embedding)
        
        # Get indices of most similar movies (excluding selected ones)
        recommendation_indices = []
        idx_sorted = np.argsort(-similarities)
        
        selected_indices = selected_indices if 'selected_indices' in locals() else []
        for idx in idx_sorted:
            if idx not in selected_indices:
                recommendation_indices.append(idx)
                if len(recommendation_indices) == n_recommendations:
                    break
        
        # Get recommended movie titles
        item_ids = self.gcl_item_ids if embedding_type == "LLM + GCL" else self.llm_item_ids
        recommended_ids = item_ids[recommendation_indices]
        recommended_movies = self.movies_df[self.movies_df['item_id'].isin(recommended_ids)]
        
        if len(recommended_movies) == 0:
            print("Warning: No matching movies found in the movies database")
            return []
        
        # Sort recommendations by similarity score
        similarity_scores = similarities[recommendation_indices]
        recommendations_with_scores = []
        for movie_id, score in zip(recommended_ids, similarity_scores):
            movie_title = self.movies_df[self.movies_df['item_id'] == movie_id]['title'].iloc[0]
            recommendations_with_scores.append((movie_title, score))
        
        recommendations_with_scores.sort(key=lambda x: x[1], reverse=True)
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
        Get personalized movie recommendations based on your taste and preferences!
        
        **How to use:**
        1. Search and select up to 5 movies you've enjoyed
        2. Describe what kind of movie you're looking for (optional)
        3. Adjust the preference weight (α) to balance between your description and movie history
        4. Get personalized recommendations
        """
        )
        
        selected_movies = gr.State([])
        retrieval_results = gr.State([])  # Store retrieval results for ranking
        
        with gr.Row():
            with gr.Column():
                # Movie search and selection
                movie_search = gr.Dropdown(
                    choices=[],
                    label="Search and select movies you've enjoyed",
                    interactive=True,
                    allow_custom_value=True
                )
                
                # Display selected movies with delete buttons
                with gr.Column(elem_id="selected_movies_container") as selected_movies_container:
                    selected_display = gr.Markdown(
                        label="Your Selected Movies",
                        value="No movies selected yet"
                    )
                    with gr.Row() as delete_row:
                        delete_buttons = []
                        for i in range(5):  # Maximum 5 movies
                            delete_buttons.append(
                                gr.Button("🗑️ Delete Movie " + str(i+1), visible=False, size="sm", min_width=100)
                            )
                
                # User preferences text field
                user_preferences = gr.Textbox(
                    label="Describe what kind of movie you're looking for",
                    placeholder="E.g., 'A thrilling sci-fi movie with deep philosophical themes'",
                    lines=3
                )
                
                # Alpha slider
                alpha = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.1,
                    label="Preference Weight (α)",
                    info="0: Use only movie history, 1: Use only your description"
                )
                
                # Embedding type selection (defaulting to GCL)
                embedding_type = gr.Radio(
                    choices=["LLM + GCL", "LLM"],
                    value="LLM + GCL",
                    label="Embedding Type",
                    info="Choose between pure language model embeddings (LLM) or graph-enhanced embeddings (LLM + GCL)"
                )
                
                # Clear selection button
                clear_btn = gr.Button("Clear Selection")
                
                # Get recommendations button
                recommend_btn = gr.Button("Get Recommendations", variant="primary")
            
            with gr.Column():
                # Display recommendations with streaming
                recommendations = gr.Markdown(
                    label="Your Personalized Recommendations",
                    value="Recommendations will appear here"
                )
        
        def update_search_options(query):
            if not query:
                return gr.Dropdown(choices=[])
            matches = recommender.search_movies(query)
            return gr.Dropdown(choices=matches)
        
        def delete_movie(btn_idx, current_movies):
            if not current_movies or btn_idx >= len(current_movies):
                button_visibility = [False] * 5
                return (
                    current_movies, 
                    format_selected_movies_with_buttons(current_movies),
                    *button_visibility  # Unpack the list of button visibilities
                )
            
            current_movies.pop(btn_idx)
            button_visibility = [i < len(current_movies) for i in range(5)]
            return (
                current_movies, 
                format_selected_movies_with_buttons(current_movies),
                *button_visibility  # Unpack the list of button visibilities
            )
        
        def format_selected_movies_with_buttons(movies):
            if not movies:
                return "No movies selected yet"
            # Format each movie with a number
            return "\n".join(f"{i+1}. {movie}" for i, movie in enumerate(movies))
        
        def add_movie(movie, current_movies):
            if not movie:
                return (
                    current_movies, 
                    format_selected_movies_with_buttons(current_movies),
                    *[i < len(current_movies) for i in range(5)]
                )
                
            current_movies = current_movies or []
            if len(current_movies) >= 5:
                return (
                    current_movies, 
                    format_selected_movies_with_buttons(current_movies),
                    *[i < len(current_movies) for i in range(5)]
                )
                
            if movie not in current_movies:
                current_movies.append(movie)
            
            # Update button visibility
            button_visibility = [i < len(current_movies) for i in range(5)]
            
            return (
                current_movies, 
                format_selected_movies_with_buttons(current_movies),
                *button_visibility
            )
        
        def clear_selection(current_movies):
            button_visibility = [False] * 5
            return [], "No movies selected yet", *button_visibility
        
        def get_recommendations(movies, emb_type, preferences, pref_weight):
            if not movies and not preferences:
                return "Please select some movies or provide preferences"
                
            # First get retrieval recommendations
            retrieval_results = recommender.get_recommendations(
                movies, 
                emb_type,
                user_preferences=preferences,
                alpha=pref_weight
            )
            
            if not retrieval_results:
                return "No recommendations available yet"
            
            # Prepare context for AI ranking
            context = {
                "user_intention": preferences if preferences else "No specific preferences provided",
                "user_history": movies if movies else "No movie history provided",
                "preference_weight": pref_weight,
                "candidates": [title for title, _ in retrieval_results]
            }
            
            # Get AI ranking and explanations with streaming
            result = "## Your Personalized Recommendations\n\n"
            yield result + "Analyzing your preferences..."
            
            for ranked_results in rank_with_ai(context):
                result = "## Your Personalized Recommendations\n\n"
                for i, (title, explanation) in enumerate(ranked_results, 1):
                    result += f"### {i}. {title}\n"
                    result += f"{explanation}\n\n"
                yield result
        
        # Event handlers
        movie_search.change(
            fn=update_search_options,
            inputs=movie_search,
            outputs=movie_search
        )
        
        movie_search.select(
            fn=add_movie,
            inputs=[movie_search, selected_movies],
            outputs=[selected_movies, selected_display] + delete_buttons
        )
        
        clear_btn.click(
            fn=clear_selection,
            inputs=[selected_movies],
            outputs=[selected_movies, selected_display] + delete_buttons
        )
        
        # Add delete button handlers
        for i, btn in enumerate(delete_buttons):
            btn.click(
                fn=delete_movie,
                inputs=[gr.Number(value=i, visible=False), selected_movies],
                outputs=[selected_movies, selected_display] + delete_buttons
            )
        
        recommend_btn.click(
            fn=get_recommendations,
            inputs=[selected_movies, embedding_type, user_preferences, alpha],
            outputs=recommendations
        )
    
    return iface

if __name__ == "__main__":
    iface = create_interface()
    if iface is not None:
        iface.launch()
    else:
        print("\nPlease fix the issues above and try again.") 