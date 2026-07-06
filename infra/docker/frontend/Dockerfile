FROM node:22-alpine AS build

WORKDIR /app

COPY apps/frontend/package.json apps/frontend/package-lock.json ./
RUN npm install --no-audit --no-fund

COPY apps/frontend/ .

ARG VITE_RECOMMENDER_API_URL=http://localhost:8090
ENV VITE_RECOMMENDER_API_URL=${VITE_RECOMMENDER_API_URL}

RUN npm run build

FROM nginx:1.27-alpine

COPY infra/docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
