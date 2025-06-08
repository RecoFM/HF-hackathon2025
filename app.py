import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import zlib
from typing import Dict, List, Tuple, Optional, Literal
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv
from ranking_agent import rank_with_ai
from scipy.sparse import load_npz
from rapidfuzz import process, fuzz
import re
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class MovieRecommender:
    def __init__(self, data_dir: str = "amazon_movies_2023"):
        self.data_dir = data_dir
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
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
        
        # Pre-process titles for fuzzy matching
        self.clean_titles = {self.clean_title_for_comparison(title): title for title in self.title_to_id.keys()}
        
    def clean_title_for_comparison(self, title):
        """Clean title for comparison purposes"""
        # Remove special characters and extra spaces
        title = re.sub(r'[^\w\s]', '', str(title))
        # Convert to lowercase and strip
        return ' '.join(title.lower().split())

    def search_movies(self, query: str) -> List[str]:
        if not query:
            return []  # Return empty if no query to avoid overwhelming UI
        
        clean_query = self.clean_title_for_comparison(query)
        # Use rapidfuzz to find matches across entire dataset
        matches = process.extract(
            clean_query,
            self.clean_titles.keys(),
            scorer=fuzz.WRatio,  # WRatio works well for movie titles
            limit=None,  # No limit - show all matches
            score_cutoff=60  # Only return matches with score > 60
        )
        
        # Convert matches back to original titles
        return [self.clean_titles[match[0]] for match in matches]
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using LangChain Mistral embeddings"""
        try:
            embedding = self.embeddings.embed_query(text)
            # Convert embedding to numpy array
            embedding = np.array(embedding, dtype=np.float32)
            # Normalize the embedding
            if np.any(embedding):  # Only normalize if not all zeros
                embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"Error getting embedding from Mistral API: {str(e)}")
            return None
        
    def get_recommendations(self, selected_movies: List[str], embedding_type: str = "LLM + GCL", user_preferences: str = "", alpha: float = 0.5) -> str:
        """
        Get recommendations using proper embedding aggregation:
        - e_h: embedding from user history (selected movies)
        - e_u: embedding from user preferences (text)
        - Combined: alpha * e_u + (1-alpha) * e_h
        """
        if not selected_movies and not user_preferences:
            return "Please select some movies or provide preferences."

        # Choose embeddings based on type
        if embedding_type == "LLM + GCL":
            embeddings = self.gcl_embeddings
            id_to_idx = self.gcl_id_to_idx
        else:
            embeddings = self.llm_embeddings
            id_to_idx = self.llm_id_to_idx

        user_profile = None
        
        # Get embedding from user history (e_h)
        e_h = None
        if selected_movies:
            movie_ids = [self.title_to_id[title] for title in selected_movies if title in self.title_to_id]
            if movie_ids:
                selected_embeddings = []
                for movie_id in movie_ids:
                    if movie_id in id_to_idx:
                        idx = id_to_idx[movie_id]
                        selected_embeddings.append(embeddings[idx])
                
                if selected_embeddings:
                    e_h = np.mean(selected_embeddings, axis=0)
        
        # Get embedding from user preferences (e_u)
        e_u = None
        if user_preferences.strip():
            e_u = self.get_text_embedding(user_preferences)
        
        # Apply aggregation algorithm
        if e_h is not None and e_u is not None:
            # Both available: alpha * e_u + (1-alpha) * e_h
            user_profile = alpha * e_u + (1 - alpha) * e_h
            print(f"Using combined embedding: α={alpha} (preferences weight)")
        elif e_u is not None:
            # Only preferences available
            user_profile = e_u
            print("Using preferences-only embedding")
        elif e_h is not None:
            # Only history available
            user_profile = e_h
            print("Using history-only embedding")
        else:
            return "Could not create user profile from provided input."
        
        # Calculate similarity with all movies
        # Normalize user profile and embeddings for proper cosine similarity
        user_profile_norm = user_profile / np.linalg.norm(user_profile)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity (normalized dot product)
        similarities = np.dot(embeddings_norm, user_profile_norm)
        
        print(f"Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")
        
        # Get top 100 most similar movies
        top_indices = np.argsort(similarities)[-100:][::-1]
        
        # Filter out selected movies and create recommendations
        seen_titles = set(selected_movies) if selected_movies else set()
        seen_clean_titles = set(self.clean_title_for_comparison(title) for title in seen_titles)
        final_recommendations = []
        
        # Get reverse mapping for the chosen embedding type
        if embedding_type == "LLM + GCL":
            idx_to_id = {idx: item_id for item_id, idx in self.gcl_id_to_idx.items()}
        else:
            idx_to_id = {idx: item_id for item_id, idx in self.llm_id_to_idx.items()}
        
        for idx in top_indices:
            if idx not in idx_to_id:
                continue
                
            item_id = idx_to_id[idx]
            
            # Find the title for this item_id
            title = None
            for t, id_ in self.title_to_id.items():
                if id_ == item_id:
                    title = t
                    break
            
            if not title:
                continue
                
            clean_title = self.clean_title_for_comparison(title)
            
            # Skip if exact title is in seen titles
            if title in seen_titles:
                continue
                
            # Skip if clean version of title is in seen titles
            if clean_title in seen_clean_titles:
                continue
                
            # Skip collections/trilogies if user has seen any part
            is_collection = False
            for seen_title in seen_titles:
                seen_clean = self.clean_title_for_comparison(seen_title)
                if seen_clean in clean_title or clean_title in seen_clean:
                    if any(marker in title.lower() for marker in ['collection', 'trilogy', 'series', 'complete']):
                        is_collection = True
                        break
            if is_collection:
                continue
            
            # Check if this is a duplicate of already recommended movie
            is_duplicate = any(
                fuzz.ratio(clean_title, self.clean_title_for_comparison(rec[0])) > 90
                for rec in final_recommendations
            )
            if is_duplicate:
                continue
                
            # Add with similarity score
            final_recommendations.append((title, similarities[idx]))
            if len(final_recommendations) >= 100:
                break
        
        if not final_recommendations:
            return "No recommendations found based on your input."
        
        return final_recommendations[:100]  # Return top 100 for ranking agent

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
        1. Search and select movies you've enjoyed (no limit!)
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
                movie_search_input = gr.Textbox(
                    label="Search movies",
                    placeholder="Type to search...",
                    interactive=True,
                    every=True
                )
                
                # Show search results as a list of clickable buttons
                search_results = gr.Radio(
                    choices=[],
                    label="Search Results",
                    interactive=True,
                    visible=True
                )
                
                # Display selected movies with functional red cross buttons
                with gr.Column(elem_id="selected_movies_container") as selected_movies_container:
                    selected_display = gr.HTML(
                        label="Your Selected Movies",
                        value="<p><i>No movies selected yet</i></p>"
                    )
                    
                    # Individual delete buttons (simpler approach)  
                    delete_buttons = []
                    for i in range(20):  # Support up to 20 movies
                        btn = gr.Button(f"× Remove Movie {i+1}", visible=False, size="sm", variant="secondary")
                        delete_buttons.append(btn)
                    
                    # Clear all button
                    clear_btn = gr.Button("Clear All", size="sm", variant="secondary")
                
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
                
                # Get recommendations button
                recommend_btn = gr.Button("Get Recommendations", variant="primary")
            
            with gr.Column():
                # Display recommendations with streaming
                recommendations = gr.Markdown(
                    label="Your Personalized Recommendations",
                    value="Recommendations will appear here"
                )
        
        def update_search_results(query):
            """Update search results based on input"""
            if not query or len(query.strip()) < 2:
                return gr.Radio(choices=[], visible=False)
            
            matches = recommender.search_movies(query)
            # Limit display to first 20 for UI performance
            display_matches = matches[:20] if len(matches) > 20 else matches
            
            if display_matches:
                return gr.Radio(choices=display_matches, visible=True)
            else:
                return gr.Radio(choices=[], visible=False)
        
        def format_selected_movies_display(movies):
            """Format selected movies with remove buttons on same line"""
            if not movies:
                return "<p><i>No movies selected yet</i></p>"
            
            html_items = []
            for i, movie in enumerate(movies):
                html_items.append(f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; 
                                padding: 8px 12px; margin: 4px 0; background-color: #f8f9fa; 
                                border-radius: 6px; border-left: 3px solid #007bff;">
                        <span style="flex-grow: 1; font-size: 14px; margin-right: 10px;">{i+1}. {movie}</span>
                    </div>
                """)
            
            return f"<div>{''.join(html_items)}</div>"

        def update_delete_buttons_visibility(movies):
            """Update visibility and labels of delete buttons"""
            button_updates = []
            for i in range(20):  # Support up to 20 movies
                if i < len(movies):
                    movie_name = movies[i][:40] + ("..." if len(movies[i]) > 40 else "")
                    button_updates.append(gr.Button(f"🗑️ {movie_name}", visible=True, size="sm", variant="secondary"))
                else:
                    button_updates.append(gr.Button(f"× Remove Movie {i+1}", visible=False, size="sm", variant="secondary"))
            
            return button_updates

        def delete_movie_by_index(index, current_movies):
            """Delete movie at specific index"""
            if not current_movies or index >= len(current_movies):
                return current_movies, format_selected_movies_display(current_movies)
            
            current_movies.pop(index)
            return current_movies, format_selected_movies_display(current_movies)

        def handle_movie_selection(selected_movie, current_movies):
            """Handle movie selection from radio buttons"""
            if not selected_movie:
                return [current_movies, format_selected_movies_display(current_movies)] + update_delete_buttons_visibility(current_movies)
            
            # Check if it's a movie title (exists in our database)
            if selected_movie in recommender.title_to_id:
                # It's a movie selection - add it to the list
                current_movies = current_movies or []
                # Remove the 5-movie limit - users can now select as many as they want
                    
                if selected_movie not in current_movies:
                    current_movies.append(selected_movie)
                
                return [current_movies, format_selected_movies_display(current_movies)] + update_delete_buttons_visibility(current_movies)
            else:
                # Not a movie from database
                return [current_movies, format_selected_movies_display(current_movies)] + update_delete_buttons_visibility(current_movies)

        def clear_all_movies():
            """Clear all selected movies"""
            empty_movies = []
            return [empty_movies, "<p><i>No movies selected yet</i></p>"] + update_delete_buttons_visibility(empty_movies)

        def get_recommendations(movies, emb_type, preferences, pref_weight):
            """Get recommendations: retrieval phase only, then delegate to ranking_agent with streaming"""
            if not movies and not preferences:
                yield "Please select some movies or provide preferences"
                return
            
            try:
                # RETRIEVAL PHASE: Get top 100 candidates using proper embedding aggregation
                print(f"\n=== RETRIEVAL PHASE ===")
                print(f"Selected movies: {movies}")
                print(f"User preferences: '{preferences}'")
                print(f"Alpha weight: {pref_weight}")
                print(f"Embedding type: {emb_type}")
                
                yield "🔍 Searching for similar movies..."
                
                recommendations = recommender.get_recommendations(
                    selected_movies=movies, 
                    embedding_type=emb_type,
                    user_preferences=preferences,
                    alpha=pref_weight
                )
                
                # Handle error cases
                if isinstance(recommendations, str):
                    yield recommendations
                    return
                
                # Print retrieval results
                print(f"\nRETRIEVAL RESULTS: Found {len(recommendations)} candidates")
                print("Top 100 from retrieval phase:")
                for i, (title, score) in enumerate(recommendations[:100], 1):
                    print(f"  {i:2d}. {title} (score: {score:.3f})")
                
                # RERANKING + EXPLANATION PHASE: Delegate to ranking_agent with streaming
                print(f"\n=== RERANKING PHASE ===")
                print(f"Calling rank_with_ai with:")
                print(f"  - {len(recommendations)} recommendations")
                print(f"  - preferences: '{preferences}'")
                print(f"  - alpha: {pref_weight}")
                print(f"  - user_movies: {movies}")
                
                yield "🤖 AI is ranking and explaining your recommendations..."
                
                # Stream the responses from ranking agent
                for partial_result in rank_with_ai(
                    recommendations=recommendations, 
                    user_preferences=preferences, 
                    alpha=pref_weight,
                    user_movies=movies
                ):
                    yield partial_result
                    
            except Exception as e:
                print(f"ERROR in get_recommendations: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"Error getting recommendations: {str(e)}"

        # Event handlers
        movie_search_input.change(
            fn=update_search_results,
            inputs=movie_search_input,
            outputs=search_results
        )
        
        search_results.change(
            fn=handle_movie_selection,
            inputs=[search_results, selected_movies],
            outputs=[selected_movies, selected_display] + delete_buttons
        )
        
        # Add individual delete button handlers
        for i, btn in enumerate(delete_buttons):
            def make_delete_handler(btn_idx):
                def delete_handler(current_movies):
                    updated_movies, updated_display = delete_movie_by_index(btn_idx, current_movies)
                    return [updated_movies, updated_display] + update_delete_buttons_visibility(updated_movies)
                return delete_handler
            
            btn.click(
                fn=make_delete_handler(i),
                inputs=[selected_movies],
                outputs=[selected_movies, selected_display] + delete_buttons
            )
        
        clear_btn.click(
            fn=clear_all_movies,
            inputs=[],
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