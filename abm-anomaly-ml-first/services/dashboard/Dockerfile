# Stage 1 - Build React app
FROM node:18 AS builder

WORKDIR /app

# Accept runtime API URL
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

COPY package.json package-lock.json ./
RUN npm ci

# Copy the entire dashboard source code
COPY . .

# Build the React app
RUN npm run build

# Stage 2 - Serve with Nginx
FROM nginx:1.25-alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# ✅ Correct: Copy from builder stage
COPY --from=build-stage /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]