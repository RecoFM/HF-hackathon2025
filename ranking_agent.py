from typing import List, Tuple, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()

def create_ranking_chain():
    """Create a ranking chain using new RunnableSequence format"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a movie recommendation expert. Your task is to select the top 10 most relevant movies from a list of recommended movies and provide the final formatted output with brief explanations.

Rules:
1. Always return exactly 10 movies
2. Consider both relevance scores and how well each movie matches user preferences
3. Pay attention to the alpha weighting parameter - it tells you how much to prioritize text preferences vs viewing history
4. Return only movies from the provided list
5. NEVER recommend movies that are already in the user's viewing history - these should be completely excluded
6. Format each movie exactly as: **1. Movie Title**\n[Exactly 2 sentences explaining why this movie matches their taste]\n\n
7. Number from 1 to 10, no additional text before or after"""),
        ("user", """Given these movie recommendations with their relevance scores:
{movie_scores}

User preferences: {preferences}

User's viewing history (DO NOT RECOMMEND ANY OF THESE): {user_movies}

Alpha weighting: {alpha}
(α=0.0 means recommendations were based entirely on viewing history, α=1.0 means entirely on text preferences, α=0.5 means equal balance)

Select the 10 most relevant movies and provide the final formatted output with explanations. Format each as:
**1. Movie Title**
[Exactly 2 sentences explaining why this movie matches their taste based on the weighted combination of their preferences and history]

**2. Movie Title**
[Exactly 2 sentences explaining why this movie matches their taste based on the weighted combination of their preferences and history]

...continue for all 10 movies.

Remember: NEVER include any movie from the user's viewing history in your recommendations.""")
    ])

    model = ChatMistralAI(
        mistral_api_key=os.environ["MISTRAL_API_KEY"],
        model="mistral-large-latest", 
        temperature=0.5,
        max_tokens=1200,
        streaming=True
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
        # Format movie scores for ranking
        movie_scores = "\n".join(
            f"{title} (relevance: {score:.3f})"
            for title, score in recommendations
        )
        
        # Start with header
        result_header = "## 🎬 Your Personalized Movie Recommendations\n\n"
        
        if user_movies and user_preferences:
            result_header += f"*Based on α={alpha} weighting: {int((1-alpha)*100)}% your viewing history + {int(alpha*100)}% your preferences*\n\n"
        elif user_preferences:
            result_header += f"*Based entirely on your preferences: \"{user_preferences}\"*\n\n"
        elif user_movies:
            result_header += f"*Based entirely on your viewing history*\n\n"
        
        result_header += "---\n\n"
        yield result_header
        
        # Single chain that does both ranking and explanation
        ranking_chain = create_ranking_chain()
        print("Calling unified ranking + explanation chain...")
        
        # Stream the response directly
        accumulated_text = result_header
        for chunk in ranking_chain.stream({
            "movie_scores": movie_scores,
            "preferences": user_preferences if user_preferences else "No specific preferences provided",
            "user_movies": ", ".join(user_movies) if user_movies else "None",
            "alpha": alpha
        }):
            if chunk.content:
                accumulated_text += chunk.content
                yield accumulated_text
            
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