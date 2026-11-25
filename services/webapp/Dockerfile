FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Empty = same origin: the page and /api come from one address, so the browser uses relative URLs.
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
# Defaults for running the image on its own; docker-compose.yml sets them explicitly.
ENV API_URL=http://api:8000 \
    WEBAPP_DNS_RESOLVER=127.0.0.11
# Rendered with envsubst at container start (API_URL, WEBAPP_DNS_RESOLVER) into /etc/nginx/conf.d.
COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=5s --retries=3 \
  CMD wget -qO /dev/null http://127.0.0.1:8080/healthz || exit 1
