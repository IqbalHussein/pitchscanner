# Frontend image: build with Node, serve the static bundle with nginx, and proxy
# /v1 to the API so the browser only ever talks to one origin.
# A glibc builder, deliberately: package-lock.json is resolved on glibc, and on
# musl npm omits @rollup/rollup-linux-x64-musl, so `vite build` fails inside the
# image while working everywhere else. The runtime below stays alpine - it only
# serves static files.
FROM node:20-bookworm-slim AS builder

WORKDIR /build
COPY package.json package-lock.json* ./
# `npm ci` with no fallback: `|| npm install` would quietly paper over a lockfile
# that no longer matches package.json, which is the thing a build should catch.
RUN npm ci
COPY . .
# The generated API types are not committed (they derive from openapi.json, which
# is), so generate them here rather than relying on a host copy being present.
RUN npm run gen:api && npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template
ENV API_UPSTREAM=http://api:8000
EXPOSE 80
