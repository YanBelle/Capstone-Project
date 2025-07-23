# Stage 1: Build React App
FROM node:18-alpine as build-stage

WORKDIR /app

# Copy package files and install deps
COPY package*.json ./
RUN npm install

# Copy source code and build the app
COPY . .
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine

# Remove default nginx index page
RUN rm -rf /usr/share/nginx/html/*

# Copy built React app
COPY --from=build-stage /app/build /usr/share/nginx/html

# Replace Nginx config
COPY default.conf /etc/nginx/conf.d/default.conf