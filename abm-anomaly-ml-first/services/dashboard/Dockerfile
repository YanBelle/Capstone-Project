# Stage 1: Build React App
FROM node:18-alpine as builder

# Set working directory
WORKDIR /app

# Copy source files
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Transfer build files only
FROM alpine:3.18

# Set target directory for Nginx to serve from
WORKDIR /nginx/html

# Copy build artifacts from the builder
COPY --from=builder /app/build ./

# This image is just for copying files during docker-compose
# You will mount this output in nginx using:
#   - ./nginx/html:/usr/share/nginx/html

EXPOSE 3000

CMD ["npx", "serve", "-s", "build", "-l", "3000"]
