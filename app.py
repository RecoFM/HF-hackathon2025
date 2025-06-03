from typing import List, Optional, Dict, Any, Union
import os
import gradio as gr
from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for the recommendation app"""
    model_name: str = "mistral-embed"
    title: str = "Foundation Recommender"
    description: str = "Get personalized foundation recommendations based on your description"
    input_label: str = "Enter description for foundation recommendation"
    output_label: str = "Recommendations"
    share: bool = False
    inbrowser: bool = True

def get_embeddings(text: str, config: Config) -> Union[List[float], str]:
    """
    Get embeddings from Mistral API
    
    Args:
        text: Input text to embed
        config: Configuration object
        
    Returns:
        List of embedding values or error message string
    """
    try:
        response = client.embeddings(
            model=config.model_name,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        return f"Error: {str(e)}"

def recommend_foundations(description: str, config: Config) -> str:
    """
    Generate foundation recommendations based on description
    
    Args:
        description: User input description
        config: Configuration object
        
    Returns:
        Recommendation results as string
    """
    # Get embeddings for the input description
    embeddings = get_embeddings(description, config)
    
    # TODO: Implement recommendation logic using embeddings
    # This is where you'll add your recommendation algorithm
    
    return "Recommendation results will appear here"

def create_interface(config: Config) -> gr.Interface:
    """
    Create Gradio interface
    
    Args:
        config: Configuration object
        
    Returns:
        Configured Gradio interface
    """
    return gr.Interface(
        fn=lambda x: recommend_foundations(x, config),
        inputs=gr.Textbox(label=config.input_label),
        outputs=gr.Textbox(label=config.output_label),
        title=config.title,
        description=config.description
    )
    
def create_gradio_interface() -> None:
    """
    Create Gradio interface and run it
    """
    movies = [
        "The Godfather", "Star Wars", "The Dark Knight", "Pulp Fiction",
        "The Lord of the Rings", "Forrest Gump", "Inception", "Fight Club",
        "The Matrix", "Goodfellas"
    ]

    def show_selected(movie_list):
        return f"You selected: {', '.join(movie_list)}"

    with gr.Blocks() as demo:
        movie_selector = gr.Dropdown(
            label="Select Movies",
            choices=movies,
            multiselect=True,
            allow_custom_value=True,  # user can also type something new
            filterable=True
        )
        output_box = gr.Textbox(label="Output", interactive=False)
        submit_btn = gr.Button("Submit")

        submit_btn.click(fn=show_selected, inputs=movie_selector, outputs=output_box)

    demo.launch()

def main() -> None:
    # # Load environment variables
    # load_dotenv()

    # # Check for API key
    # api_key = os.getenv("MISTRAL_API_KEY")
    # if not api_key:
    #     raise ValueError("MISTRAL_API_KEY not found in environment variables")

    # # Initialize Mistral client
    # global client
    # client = MistralClient(api_key=api_key)
    
    # # Initialize configuration
    # config = Config()
    
    # # Create and launch interface
    # interface = create_interface(config)
    # interface.launch(share=config.share, inbrowser=config.inbrowser)
    create_gradio_interface()

if __name__ == "__main__":
    main() 