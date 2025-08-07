FROM node:20

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl wget ca-certificates gnupg

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of your code
COPY . .



EXPOSE 3001

# Start Ollama in the background, pull the model, then start Node.js app
CMD ollama serve & \
  bash -c 'until curl -s http://localhost:11434; do sleep 1; done; ollama pull nomic-embed-text:latest' && \
  node index.js
