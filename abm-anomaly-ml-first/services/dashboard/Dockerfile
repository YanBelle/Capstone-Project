# Stage 1 - Build React Dashboard
FROM node:18 AS builder

WORKDIR /app

# Accept runtime variable for API URL
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

# Copy and install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the source code
COPY . .

# Build the React app with API URL injected
RUN npm run build

# Stage 2 - Serve with Nginx
FROM nginx:1.25-alpine

# Remove default config and copy your custom config
RUN rm /etc/nginx/conf.d/default.conf
COPY ./dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy build files
COPY --from=builder /app/build /usr/share/nginx/html

# Expose default port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]