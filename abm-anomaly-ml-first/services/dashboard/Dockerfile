FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM node:18-alpine

WORKDIR /app

# To run in Nginx
COPY --from=builder /app/build /home/deploy/Capstone-Project/abm-anomaly-ml-first/nginx/html
COPY --from=builder /app/package*.json /home/deploy/Capstone-Project/abm-anomaly-ml-first/nginx/html

# To run with serve
# Uncomment the following lines if you want to use serve instead of Nginx
#COPY --from=builder /app/build ./build
#COPY --from=builder /app/package*.json ./

RUN npm install --production

EXPOSE 3000

CMD ["npx", "serve", "-s", "build", "-l", "3000"]
