FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl wget ca-certificates gnupg nodejs npm

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of your code
COPY . .

# Download embedding model for Ollama
RUN ollama pull nomic-embed-text:latest


EXPOSE 3001

# Start Ollama in the background, then Node.js app
CMD ollama serve & sleep 5 && node index.js
