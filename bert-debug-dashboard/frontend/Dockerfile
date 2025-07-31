FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Create .env file with WebSocket disabled settings
RUN echo "WDS_SOCKET_PORT=0" > .env && \
    echo "CHOKIDAR_USEPOLLING=true" >> .env && \
    echo "WATCHPACK_POLLING=true" >> .env && \
    echo "BROWSER=none" >> .env && \
    echo "REACT_APP_API_URL=http://localhost:8000" >> .env && \
    echo "GENERATE_SOURCEMAP=false" >> .env && \
    echo "DISABLE_ESLINT_PLUGIN=true" >> .env

# Build the app (if needed)
# RUN npm run build

# Use development server
EXPOSE 3000

CMD ["./start.sh"]
