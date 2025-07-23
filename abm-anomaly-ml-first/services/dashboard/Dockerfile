# ---- Stage 1: Build the React app ----
FROM node:18-alpine AS build-stage

WORKDIR /app

# Copy and install dependencies
COPY package*.json ./
RUN npm ci

# Copy app source and build
COPY . .
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build

# ---- Stage 2: Serve with nginx ----
FROM nginx:1.25-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY default.conf /etc/nginx/conf.d/

# Copy the built app from previous stage
COPY --from=build-stage /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]