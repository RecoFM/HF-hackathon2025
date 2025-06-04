# Movie Recommender System

A hybrid movie recommender system that combines collaborative filtering, language model embeddings, and graph convolutional networks to provide personalized movie recommendations.

## Features

- **Dual Embedding Types:**
  - Pure Language Model (LLM) embeddings from Mistral AI
  - Graph-enhanced embeddings (LLM + GCL) that combine language understanding with user interaction patterns
- **Hybrid Input:**
  - Select up to 5 movies you've enjoyed
  - Describe what kind of movie you're looking for in natural language
  - Adjust the weight (α) between your movie selections and text description
- **Rich Results:**
  - Get up to 20 personalized recommendations
  - View similarity scores for each recommendation
  - Search through a database of over 100,000 movies

## Requirements

1. Python 3.8+
2. Virtual environment (recommended)
3. Mistral AI API key (get one at https://console.mistral.ai/)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the project root:
```bash
MISTRAL_API_KEY=your_api_key_here
```

2. Ensure you have the necessary data files in the `amazon_movies_2023` directory:
   - `title_embeddings.npz`: Movie title embeddings from Mistral AI
   - `gcl_embeddings.npz`: Graph-enhanced embeddings
   - `title_embeddings_mapping.csv`: Movie metadata mapping

## Usage

1. Activate your virtual environment:
```bash
source venv/bin/activate  # On Unix/macOS
```

2. Run the recommender app:
```bash
python movie_recommender_app.py
```

3. Open your browser to the local URL shown in the terminal (typically http://127.0.0.1:7860)

## How It Works

1. **Movie Selection:**
   - Search and select up to 5 movies you've enjoyed
   - The system uses these as a baseline for your taste

2. **Text Preferences:**
   - Describe what you're looking for (e.g., "A thrilling sci-fi movie with deep philosophical themes")
   - Your description is converted to embeddings using Mistral AI

3. **Preference Weighting:**
   - Use the α slider to balance between your selected movies and text description
   - α = 0: Only use movie history
   - α = 1: Only use text description
   - Values in between combine both signals

4. **Embedding Types:**
   - LLM: Pure language model embeddings for semantic understanding
   - LLM + GCL: Graph-enhanced embeddings that also consider user interaction patterns

## Data Processing

For information about the dataset processing pipeline, see [DATA_PROCESSING.md](DATA_PROCESSING.md)

## Contributing

Feel free to open issues or submit pull requests with improvements! 