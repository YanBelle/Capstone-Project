# services/dashboard/Dockerfile
FROM node:18-alpine as build-stage

WORKDIR /app

# Install deps
COPY package.json package-lock.json ./
RUN npm ci

# Copy source
COPY . .

# Inject REACT_APP_API_URL from build arg and build the app
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN echo "Using API: $REACT_APP_API_URL" && npm build

# ------------------------

FROM nginx:1.25-alpine

COPY --from=build-stage /app/build /usr/share/nginx/html
COPY default.conf /etc/nginx/conf.d/default.conf