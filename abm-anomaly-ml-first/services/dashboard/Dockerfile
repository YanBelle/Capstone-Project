# --------------------------------------
# 🛠️ Build Stage
# --------------------------------------
FROM node:18-alpine3.19 AS build-stage

WORKDIR /app

# Update apk and install dependencies
RUN apk update && apk upgrade --no-cache

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy the rest of the source code
COPY . .

# Build the React app for production
RUN npm run build


# --------------------------------------
# 🚀 Production Stage
# --------------------------------------
FROM nginx:alpine AS production

# Update packages to reduce vulnerabilities and remove cache
RUN apk update && apk upgrade --no-cache

# Copy built React static files into Nginx's public folder
COPY --from=build-stage /app/build /usr/share/nginx/html

# Replace default Nginx config if needed
# COPY nginx.conf /etc/nginx/nginx.conf
# COPY default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]