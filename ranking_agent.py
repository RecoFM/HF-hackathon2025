from typing import List, Dict, Tuple
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
- Be concise but insightful (1-2 sentences per movie)

IMPORTANT REQUIREMENTS:
- You MUST provide EXACTLY 20 recommendations from the candidate list
- Number each recommendation from 1 to 20
- Keep explanations concise to maintain a good pace
- Focus on personal relevance over general movie quality
- Be specific in your explanations, avoiding generic statements
- Maintain a warm, conversational tone while being informative"""

def create_chain():
    """Create a LangChain chain for movie recommendations"""
    # Initialize the LLM
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        streaming=True,
        temperature=0.7
    )
    
    # Create the chain
    chain = (
        RunnablePassthrough()
        | {
            "system": lambda _: SYSTEM_PROMPT,
            "user": lambda x: f"""Please rerank and provide explanations for EXACTLY 20 movies from the following candidates, based on this context:

User's Intention: {x['user_intention']}

Movie History: {x['user_history']}

Preference Weight (α): {x['preference_weight']}
(0 = focus on history, 1 = focus on stated preferences)

Candidate Movies:
{chr(10).join(f'- {movie}' for movie in x['candidates'])}

IMPORTANT: You MUST provide EXACTLY 20 recommendations, numbered from 1 to 20. For each movie, start with the number, then the title, then a brief 1-2 sentence explanation of why it's a good match."""
        }
        | (lambda x: [SystemMessage(content=x["system"]), HumanMessage(content=x["user"])])
        | llm
        | StrOutputParser()
    )
    
    return chain

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
    # Ensure we have enough candidates
    if len(context['candidates']) < 20:
        print(f"Warning: Only {len(context['candidates'])} candidates available")
        # Duplicate some movies if needed to reach 20
        while len(context['candidates']) < 20:
            context['candidates'].extend(context['candidates'][:20 - len(context['candidates'])])
    
    # Create the chain if not already created
    chain = create_chain()
    
    # Parse response and extract recommendations with explanations
    recommendations = []
    current_movie = None
    current_explanation = []
    buffer = []
    
    try:
        # Stream the response
        for chunk in chain.stream(context):
            buffer.append(chunk)
            current_text = ''.join(buffer)
            
            # Check if we have complete lines
            if '\n' in chunk:
                lines = current_text.split('\n')
                buffer = [lines[-1]]  # Keep the incomplete line in buffer
                
                for line in lines[:-1]:  # Process complete lines
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if this is a new movie entry (1-20)
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
        try:
            # Create a non-streaming chain
            llm = ChatMistralAI(
                model="mistral-large-latest",
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                streaming=False,
                temperature=0.7
            )
            
            chain = (
                RunnablePassthrough()
                | {
                    "system": lambda _: SYSTEM_PROMPT,
                    "user": lambda x: f"""Please rerank and provide explanations for EXACTLY 20 movies from the following candidates, based on this context:

User's Intention: {x['user_intention']}

Movie History: {x['user_history']}

Preference Weight (α): {x['preference_weight']}
(0 = focus on history, 1 = focus on stated preferences)

Candidate Movies:
{chr(10).join(f'- {movie}' for movie in x['candidates'])}

IMPORTANT: You MUST provide EXACTLY 20 recommendations, numbered from 1 to 20. For each movie, start with the number, then the title, then a brief 1-2 sentence explanation of why it's a good match."""
                }
                | (lambda x: [SystemMessage(content=x["system"]), HumanMessage(content=x["user"])])
                | llm
                | StrOutputParser()
            )
            
            # Get complete response
            text = chain.invoke(context)
            
            # Process the complete response
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
            
        except Exception as e:
            print(f"Error in fallback mode: {str(e)}")
            yield []
    
    # If we somehow got more than 20 recommendations, trim the list
    return recommendations[:20] 