# Tomoro RAG System

This project implements a Retrieval-Augmented Generation (RAG) system with evaluation capabilities. It uses FastAPI for the API layer and Poetry for dependency management.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Setup Instructions

1. **Clone the repository** (if you haven't already):

   ```
   git clone https://github.com/your-username/tomoro.git
   cd tomoro
   ```

2. **Install Poetry**:
   If you don't have Poetry installed, run:

   ```
   pip install poetry
   ```

3. **Install project dependencies**:
   In the project root directory, run:

   ```
   poetry install
   ```

   This will create a virtual environment and install all required dependencies.

4. **Set up environment variables**:
   Create a `.env` file in the project root directory with the following content:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   DB_NAME="embeddings.db"
   EMBEDDING_MODEL="text-embedding-ada-002"
   CHAT_MODEL="gpt-4o-mini"
   DATA_PATH="path/to/your/train.json"

   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

5. **Export Python Path**:
   Run the following command to export python pathe:
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   ```

6. **Download Database**:
   You can download the embedding.db from this google drive link https://drive.google.com/file/d/1L0gwXq0OnWibkGpD7tWhptq3HnAB_nZE/view?usp=sharing


## Running the Application

1. **Start the FastAPI server**:

   ```
   poetry run python src/main.py
   ```

   The server will start running on `http://localhost:8000`.

2. **Access the API documentation**:
   Open a web browser and go to `http://localhost:8000/docs` to see the Swagger UI with API documentation.

## Using the API

The API provides two main endpoints:

1. **/rag** (POST): Perform RAG on a given query

   - Request body: `{"query": "Your question here"}`
   - Response: RAG result including the question, answer, and relevant contexts

2. **/evaluate** (POST): Evaluate the RAG system using benchmark data
   - No request body required
   - Response: Evaluation metrics

You can use these endpoints via the Swagger UI or by sending HTTP requests to the server.

## Running Tests

To run the test suite:

```
poetry run pytest
```

This will execute all the tests in the project however, there are no tests yet ;)


## Algorithm Structure
### Method 1 (Hybrid Embedding Method)

 <img src="https://github.com/mehdihosseinimoghadam/tomoro/blob/main/Algo1.png" height="600" width="940" >


### Method 2 (Knowledge Graph)


 <img src="https://github.com/mehdihosseinimoghadam/tomoro/blob/main/knowledge%20graph.png" height="600" width="940" >

## Project Structure

# Tomoro

Tomoro is a project designed to implement a Retrieval-Augmented Generation (RAG) system with embedded APIs for data retrieval and a core framework for evaluation and embeddings. The system is built using Python and follows a modular architecture with separate components for API routes, core logic, utilities, and tests.

## Project Structure

```bash
tomoro/
├── .env               # Environment variables for the project
├── .gitignore         # Git ignore file to specify ignored files and directories
├── README.md          # Project documentation
├── pyproject.toml     # Poetry configuration file for project dependencies
├── poetry.lock        # Lock file for project dependencies
├── openapi.json       # OpenAPI specification for API documentation
├── embeddings.db      # Database file for embeddings (could be SQLite or other formats)
├── src/               # Main source code directory
│   ├── __init__.py    # Marks the directory as a Python package
│   ├── main.py        # Entry point of the application
│   ├── config.py      # Configuration management for the project
│   ├── api/           # API layer to expose routes
│   │   ├── __init__.py
│   │   ├── routes.py  # API routes definitions
│   │   ├── models.py  # Data models for API requests and responses
│   ├── core/          # Core components of the RAG system
│   │   ├── __init__.py
│   │   ├── database.py    # Database interactions and ORM definitions
│   │   ├── embeddings.py  # Embeddings logic for the system
│   │   ├── rag.py         # RAG system logic
│   │   ├── evaluation.py  # Evaluation metrics and tools
│   ├── utils/         # Utility functions for common operations
│   │   ├── __init__.py
│   │   ├── helpers.py # Helper functions
├── tests/             # Unit and integration tests
    ├── __init__.py
    ├── test_api.py        # Tests for API routes and functionality
    ├── test_rag.py        # Tests for RAG functionality
    └── test_evaluation.py # Tests for evaluation metrics and tools
```

## Customization

- To modify the RAG system behavior, edit files in `src/core/`.
- To change API endpoints or add new ones, modify `src/api/routes.py`.
- To update data models, edit `src/api/models.py`.

## Troubleshooting

1. **Database issues**: If you encounter database-related errors, ensure that the `embeddings.db` file exists in the project root. 

2. **OpenAI API errors**: Verify that your OpenAI API key in the `.env` file is correct and has the necessary permissions.

3. **Dependency issues**: If you encounter dependency-related errors, try updating your dependencies with `poetry update`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```
