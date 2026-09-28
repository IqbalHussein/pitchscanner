# Frontend image: build with Node, serve the static bundle with nginx, and proxy
# /v1 to the API so the browser only ever talks to one origin.
FROM node:20-alpine AS builder

WORKDIR /build
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template
ENV API_UPSTREAM=http://api:8000
EXPOSE 80
