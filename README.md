# Better Local RAG

A better Retrieval Augmented Generation pipeline that involves chat history and prompt reformulation before retrieval for better results.

<p align="center">
    <img src="./rag-diagram.jpg" width=700/>
</p>

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ill-usion/better-local-rag.git
    cd better-local-rag
    ```

2. Create a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1. Configure the `options.yaml` file with your settings.
1. Create a `.env` file in the root directory and add your API key:
    ```
    LLM_API_KEY=<your key>
    ```

### Usage

1. Run the embedding script to initialize the vector store:

    ```bash
    python embed.py
    ```

1. Start the main application:
    ```bash
    python app.py
    ```

## Resources

- https://medium.com/keeping-up-with-ai/how-i-built-a-rag-based-ai-chatbot-from-my-personal-data-88eec0d3483c
- https://medium.com/@mr.murga/streaming-ai-responses-with-flask-a-practical-guide-677c15e82cdd
