# Nova.AI

Nova.AI is a Python-based application designed to translate natural language into SQL queries. It leverages Ollama LLM and embeddings, connects to PostgreSQL databases, fetches DDL statements, and utilizes ChromaDB as a vector database for embedding relevant data. The project also includes web scraping capabilities using the tool API.

## Features

- **Natural Language to SQL**: Translate natural language queries into SQL.
- **Database Connectivity**: Connect to PostgreSQL databases and fetch DDL statements.
- **Vector Database**: Use ChromaDB for embedding relevant data.
- **Web Scraping**: Scrape web data using the tool API.
- **Docker Support**: Easily deployable using Docker.
- **Version Control**: Managed with GitHub.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/RodolfoMontenegro/Nova.AI.git
    cd Nova.AI
    ```

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    apt-get install poppler-utils
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm
    ```

4. Set up Docker (if required):
    ```bash
    docker-compose up
    ```

## Usage

To run the main application, execute:
```bash
python mapa.py
 ```

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code adheres to the project’s coding standards and includes appropriate tests.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contact
For any questions or suggestions, feel free to open an issue or contact Rodolfo Montenegro Ochoa.
