#!/bin/bash

PROJECT_NAME="market-mind-ai"

# Create the internal folder structure
mkdir -p agents tools data prompts

# Create the core Python files
touch app.py
touch requirements.txt
touch .env
touch .gitignore
touch README.md

# Initialize the sub-packages
touch agents/__init__.py
touch agents/researcher.py
touch agents/analyst.py

touch tools/__init__.py
touch tools/stock_tools.py

# Initialize prompt templates
touch prompts/system_prompts.yaml
touch prompts/reasoning_logic.txt

# Add standard ignores to .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".streamlit/" >> .gitignore

# Add basic dependencies to requirements.txt
echo "streamlit" >> requirements.txt
echo "langchain" >> requirements.txt
echo "langchain-openai" >> requirements.txt
echo "yfinance" >> requirements.txt
echo "python-dotenv" >> requirements.txt
echo "tavily-python" >> requirements.txt

echo "✅ Project structure for '$PROJECT_NAME' created successfully!"