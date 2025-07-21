# Stage 1: Build React app
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

# Ensure output is relative to /dashboard path for Nginx routing
ENV PUBLIC_URL=/dashboard

RUN npm run build

# Stage 2: Only export built static files
FROM nginx:stable-alpine AS dashboard

# Clean default nginx site and copy built assets
RUN rm -rf /usr/share/nginx/html/*

COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# You don't need to expose or start Nginx here since this container's content is used by your reverse proxy