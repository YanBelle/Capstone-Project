#!/bin/sh

# Disable WebSocket for hot reload
export WDS_SOCKET_PORT=0
export CHOKIDAR_USEPOLLING=true
export WATCHPACK_POLLING=true
export BROWSER=none
export REACT_APP_API_URL=http://localhost:8000
export FAST_REFRESH=false
export WDS_SOCKET_HOST=localhost
export WDS_SOCKET_PATH=/ws
export GENERATE_SOURCEMAP=false
export DISABLE_ESLINT_PLUGIN=true

# Create .env file to ensure settings persist
echo "WDS_SOCKET_PORT=0" > .env
echo "CHOKIDAR_USEPOLLING=true" >> .env
echo "WATCHPACK_POLLING=true" >> .env
echo "BROWSER=none" >> .env
echo "REACT_APP_API_URL=http://localhost:8000" >> .env
echo "FAST_REFRESH=false" >> .env
echo "WDS_SOCKET_HOST=localhost" >> .env
echo "WDS_SOCKET_PATH=/ws" >> .env
echo "GENERATE_SOURCEMAP=false" >> .env
echo "DISABLE_ESLINT_PLUGIN=true" >> .env

# Start the React development server
npm start
