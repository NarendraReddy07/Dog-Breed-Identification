# Build the React frontend
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies using the committed lockfile.
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build production assets.
COPY frontend/ ./
RUN npm run build

# Serve the production build with Nginx.
FROM nginx:alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]