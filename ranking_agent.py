from typing import List, Dict, Tuple
from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert movie recommendation agent with deep knowledge of cinema, storytelling, and human psychology. Your role is to thoughtfully rerank and explain movie recommendations based on user context.

Your task has three key aspects:

1. UNDERSTAND THE CONTEXT:
- User's stated intentions/preferences
- Their movie viewing history
- The balance (α) between explicit preferences and history
   - α=0: Focus entirely on historical patterns
   - α=1: Focus entirely on stated preferences
   - Values between 0-1: Blend both signals

2. ANALYZE EACH CANDIDATE:
- Consider how well it matches user's explicit preferences
- Look for patterns in their viewing history
- Think about both obvious and subtle connections
- Consider the movie's themes, style, mood, and emotional resonance

3. PROVIDE THOUGHTFUL EXPLANATIONS:
- Explain why each movie is personally relevant
- Connect recommendations to user's preferences and history
- Highlight specific aspects that make it a good fit
- Be concise but insightful (2-3 sentences per movie)

Remember:
- Focus on personal relevance over general movie quality
- Consider both content similarity and emotional resonance
- Be specific in your explanations, avoiding generic statements
- Maintain a warm, conversational tone while being informative

Your output should be a reranked list of movies, each with a personalized explanation of why it's a good match for this specific user in this specific context."""

def rank_with_ai(context: Dict) -> List[Tuple[str, str]]:
    """
    Rerank recommendations using AI and provide explanations.
    
    Args:
        context: Dictionary containing:
            - user_intention: Stated preferences/what they're looking for
            - user_history: List of movies they've enjoyed
            - preference_weight: Alpha value for balancing preferences vs history
            - candidates: List of candidate movies to rank
            
    Returns:
        List of (movie_title, explanation) tuples in ranked order
    """
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    # Prepare the prompt
    user_message = f"""Please rerank these movie recommendations based on the following context:

User's Intention: {context['user_intention']}

Movie History: {context['user_history']}

Preference Weight (α): {context['preference_weight']}
(0 = focus on history, 1 = focus on stated preferences)

Candidate Movies:
{chr(10).join(f'- {movie}' for movie in context['candidates'])}

Please provide the reranked list with brief, personalized explanations for why each movie is a good match. For each movie, start with the number, then the title, then the explanation."""

    # Get AI response with streaming
    stream_response = client.chat.stream(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    
    # Parse response and extract recommendations with explanations
    recommendations = []
    current_movie = None
    current_explanation = []
    buffer = []
    
    try:
        for chunk in stream_response:
            if chunk.data.choices[0].delta.content:
                content = chunk.data.choices[0].delta.content
                buffer.append(content)
                current_text = ''.join(buffer)
                
                # Check if we have a complete line
                if '\n' in content:
                    lines = current_text.split('\n')
                    buffer = [lines[-1]]  # Keep the incomplete line in buffer
                    
                    for line in lines[:-1]:  # Process complete lines
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check if this is a new movie entry
                        if any(line.startswith(f"{i}.") for i in range(1, 21)):
                            # Save previous movie if exists
                            if current_movie and current_explanation:
                                recommendations.append((current_movie, ' '.join(current_explanation)))
                                yield recommendations
                            
                            # Start new movie
                            try:
                                # Extract movie title (everything between the number and the next sentence)
                                parts = line.split('.', 1)
                                if len(parts) > 1:
                                    title_and_explanation = parts[1].strip()
                                    # Find the first sentence boundary after the title
                                    sentences = title_and_explanation.split('. ')
                                    current_movie = sentences[0].strip()
                                    if len(sentences) > 1:
                                        current_explanation = ['. '.join(sentences[1:])]
                                    else:
                                        current_explanation = []
                            except Exception as e:
                                print(f"Error parsing movie line: {str(e)}")
                                continue
                        
                        # If we have a current movie, add to its explanation
                        elif current_movie and line:
                            current_explanation.append(line)
                            yield recommendations + [(current_movie, ' '.join(current_explanation))]
        
        # Add last movie
        if current_movie and current_explanation:
            recommendations.append((current_movie, ' '.join(current_explanation)))
            yield recommendations
            
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        # If streaming fails, fall back to non-streaming completion
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Process the complete response
        text = response.choices[0].message.content
        current_movie = None
        current_explanation = []
        recommendations = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if any(line.startswith(f"{i}.") for i in range(1, 21)):
                if current_movie and current_explanation:
                    recommendations.append((current_movie, ' '.join(current_explanation)))
                
                try:
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        title_and_explanation = parts[1].strip()
                        sentences = title_and_explanation.split('. ')
                        current_movie = sentences[0].strip()
                        if len(sentences) > 1:
                            current_explanation = ['. '.join(sentences[1:])]
                        else:
                            current_explanation = []
                except Exception as e:
                    print(f"Error parsing movie line: {str(e)}")
                    continue
            elif current_movie and line:
                current_explanation.append(line)
        
        if current_movie and current_explanation:
            recommendations.append((current_movie, ' '.join(current_explanation)))
        
        yield recommendations
    
    return recommendations[:20]  # Ensure we return at most 20 recommendations 