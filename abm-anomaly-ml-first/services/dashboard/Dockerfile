# Stage 1 - Build React Dashboard
FROM node:18 AS builder

# Set working directory inside container
WORKDIR /app

# Accept runtime variable for API URL (for use in React build)
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the dashboard source
COPY . .

# Build the React app
RUN npm run build

# Stage 2 - Serve with Nginx
FROM nginx:1.25-alpine

# Remove default Nginx site config and replace with our own
RUN rm /etc/nginx/conf.d/default.conf
COPY default.conf /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built React app from previous stage
#COPY --from=builder /app/build /usr/share/nginx/html
COPY ./build /usr/share/nginx/html

# Expose default HTTP port
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]