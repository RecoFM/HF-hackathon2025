from typing import List, Tuple, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()

def create_ranking_chain():
    """Create a ranking chain using new RunnableSequence format"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a movie recommendation expert. Your task is to select the top 10 most relevant movies from a list of recommended movies, considering both their relevance scores and user preferences.

Rules:
1. Always return exactly 10 movies
2. Consider both relevance scores and how well each movie matches user preferences
3. Pay attention to the alpha weighting parameter - it tells you how much to prioritize text preferences vs viewing history
4. Return only movies from the provided list
5. Format output as a simple list of movie titles, one per line
6. Do not include numbers, explanations, or any other text"""),
        ("user", """Given these movie recommendations with their relevance scores:
{movie_scores}

User preferences: {preferences}

Alpha weighting: {alpha}
(α=0.0 means recommendations were based entirely on viewing history, α=1.0 means entirely on text preferences, α=0.5 means equal balance)

Select the 10 most relevant movies for this user, considering both the relevance scores and how the alpha weighting affects what the user values most.""")
    ])

    model = ChatMistralAI(
        mistral_api_key=os.environ["MISTRAL_API_KEY"],
        model="mistral-small", 
        temperature=0.3,
        max_tokens=500
    )

    return prompt | model

def create_explanation_chain():
    """Create an explanation chain using new RunnableSequence format"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a movie expert who provides personalized, concise explanations for movie recommendations. Create engaging explanations that connect the user's preferences and viewing history to each recommended movie. Keep each explanation to exactly 2 sentences maximum. Be conversational and insightful.

IMPORTANT: Do not provide any introduction or summary text. Start directly with the numbered recommendations."""),
        ("user", """Based on the user's movie preferences: "{preferences}"
And their viewing history: {movies}
Preference weight (α): {alpha} (where α=0 means only history matters, α=1 means only preferences matter, α=0.5 means equal weight)

Provide explanations for these 10 movies, showing why each one suits the user's taste (even if partially):
{recommendations}

Format each recommendation as:
**1. Movie Title**
**2. Movie Title**
...
**10. Movie Title**

[Exactly 2 sentences explaining why this movie matches their taste based on the weighted combination of their preferences and history. Always find positive connections, even if partial.]

Start directly with the first recommendation. No introduction text.""")
    ])

    model = ChatMistralAI(
        mistral_api_key=os.environ["MISTRAL_API_KEY"],
        model="mistral-large-latest",
        temperature=0.7,
        max_tokens=800,
        streaming=True  # Enable streaming
    )

    return prompt | model

def rank_with_ai(recommendations: List[Tuple[str, float]], user_preferences: str = "", alpha: float = 0.5, user_movies: List[str] = None):
    """
    Complete reranking and explanation pipeline with streaming:
    1. Takes top 100 candidates from retrieval phase
    2. Reranks to top 10 using AI
    3. Generates explanations with streaming
    4. Yields partial formatted responses
    
    Args:
        recommendations: List of (movie_title, relevance_score) tuples from retrieval phase
        user_preferences: User's textual preferences/description
        alpha: Weighting parameter (0.0 = only history matters, 1.0 = only preferences matter)
        user_movies: List of user's selected movies for context
    """
    print(f"\n=== RANKING_AGENT DEBUG ===")
    print(f"Received {len(recommendations) if recommendations else 0} recommendations")
    print(f"User preferences: '{user_preferences}' (length: {len(user_preferences) if user_preferences else 0})")
    print(f"Alpha: {alpha}")
    print(f"User movies: {user_movies}")
    
    if not recommendations:
        yield "No recommendations available."
        return
    
    # Take only top 100 recommendations if more are provided
    recommendations = recommendations[:100]
    
    try:
        # Step 1: Always get exactly 10 movies (rerank if preferences, otherwise top 10 by score)
        if user_preferences and user_preferences.strip():
            print("=== STEP 1: RERANKING WITH PREFERENCES ===")
            # Format movie scores for ranking
            movie_scores = "\n".join(
                f"{title} (relevance: {score:.3f})"
                for title, score in recommendations
            )
            
            # Get top 10 ranked movies
            ranking_chain = create_ranking_chain()
            print("Calling ranking chain...")
            ranking_response = ranking_chain.invoke({
                "movie_scores": movie_scores,
                "preferences": user_preferences,
                "alpha": alpha
            })
            
            print(f"Ranking response: {ranking_response.content}")
            
            # Extract movie titles from ranking response
            selected_movies = []
            titles = [title for title, _ in recommendations]
            
            for line in ranking_response.content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                title = line.strip('.- ')
                if title in titles:
                    selected_movies.append(title)
                    if len(selected_movies) >= 10:
                        break
            
            # ALWAYS ensure we have exactly 10 movies - this is CRITICAL
            if len(selected_movies) < 10:
                remaining = set(titles) - set(selected_movies)
                remaining_sorted = sorted(
                    [(title, score) for title, score in recommendations if title in remaining],
                    key=lambda x: x[1],
                    reverse=True
                )
                needed = 10 - len(selected_movies)
                selected_movies.extend(title for title, _ in remaining_sorted[:needed])
                print(f"Added {needed} movies to reach exactly 10: {selected_movies[-needed:]}")
            
            # Final safety check - if still not 10, use top movies by score
            if len(selected_movies) < 10:
                print(f"Warning: Still only have {len(selected_movies)} movies, filling with top scores")
                top_by_score = [title for title, _ in recommendations[:15]]  # Take more to account for duplicates
                for title in top_by_score:
                    if title not in selected_movies:
                        selected_movies.append(title)
                        if len(selected_movies) >= 10:
                            break
            
            top_10_movies = selected_movies[:10]  # Guarantee exactly 10
            print(f"FINAL: Exactly {len(top_10_movies)} movies selected: {top_10_movies}")
        else:
            print("=== STEP 1: NO PREFERENCES - USING SCORE RANKING ===")
            # No preferences, just take top 10 by score
            top_10_movies = [title for title, _ in recommendations[:10]]
            print(f"Top 10 by score: {top_10_movies}")
        
        # Step 2: Generate explanations with streaming
        if user_preferences and user_preferences.strip():
            print("=== STEP 2: GENERATING EXPLANATIONS WITH STREAMING ===")
            
            # Start with header
            result_header = "## 🎬 Your Personalized Movie Recommendations\n\n"
            
            if user_movies and user_preferences:
                result_header += f"*Based on α={alpha} weighting: {int((1-alpha)*100)}% your viewing history + {int(alpha*100)}% your preferences*\n\n"
            elif user_preferences:
                result_header += f"*Based entirely on your preferences: \"{user_preferences}\"*\n\n"
            else:
                result_header += f"*Based entirely on your viewing history*\n\n"
            
            result_header += "---\n\n"
            yield result_header
            
            # Create explanation chain with streaming
            explanation_chain = create_explanation_chain()
            
            explanation_prompt = {
                "preferences": user_preferences,
                "movies": ", ".join(user_movies) if user_movies else "None",
                "alpha": alpha,
                "recommendations": "\n".join(f"{i+1}. {title}" for i, title in enumerate(top_10_movies))
            }
            print(f"Explanation prompt data: {explanation_prompt}")
            
            # Stream the response
            accumulated_text = result_header
            for chunk in explanation_chain.stream(explanation_prompt):
                if chunk.content:
                    accumulated_text += chunk.content
                    yield accumulated_text
                    
        else:
            print("=== STEP 2: NO PREFERENCES - SIMPLE FORMAT ===")
            # Simple format without preferences
            result = "## 🎬 Movies You Might Like\n\n"
            if user_movies:
                result += f"*Based on your viewing history: {', '.join(user_movies)}*\n\n"
            
            # Get scores for display
            movie_scores = {title: score for title, score in recommendations}
            for i, title in enumerate(top_10_movies, 1):
                score = movie_scores.get(title, 0.0)
                result += f"**{i}. {title}**\n"
                result += f"*Similarity: {score:.3f}*\n\n"
            
            yield result
            
    except Exception as e:
        print(f"ERROR in rank_with_ai: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to simple format
        result = "## 🎬 Your Recommendations\n\n"
        for i, (title, score) in enumerate(recommendations[:10], 1):
            result += f"**{i}. {title}**\n"
            result += f"*Similarity: {score:.3f}*\n\n"
        yield result 