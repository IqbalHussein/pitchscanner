# Frontend image: build with Node, serve the static bundle with nginx, and proxy
# /v1 to the API so the browser only ever talks to one origin.
FROM node:20-alpine AS builder

WORKDIR /build
COPY package.json package-lock.json* ./
# `npm ci` with no fallback: `|| npm install` would quietly paper over a lockfile
# that no longer matches package.json, which is the thing a build should catch.
# It resolves rollup's musl binding correctly here - the lockfile carries the musl
# optional dependencies - as long as .dockerignore keeps the host's glibc
# node_modules out of the image. Without that, `vite build` fails with "Cannot
# find module @rollup/rollup-linux-x64-musl".
# The `test -x` is not paranoia: npm's "Exit handler never called!" crash - which
# a flaky registry connection triggers reliably - can leave a partial node_modules
# behind *and* exit 0, and the build then fails two steps later with a confusing
# "openapi-typescript: not found". Fail here, where the cause is visible.
RUN npm ci --no-audit --no-fund --fetch-retries=5 --fetch-retry-maxtimeout=120000 \
 && test -x node_modules/.bin/openapi-typescript \
 && test -x node_modules/.bin/vite
COPY . .
# The generated API types are not committed (they derive from openapi.json, which
# is), so generate them here rather than relying on a host copy being present.
RUN npm run gen:api && npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template
ENV API_UPSTREAM=http://api:8000
EXPOSE 80
