# Un seul conteneur : le front est compilé puis servi par l'API (voir server/index.js).

# --- 1. Compilation du front -----------------------------------------
FROM node:22-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- 2. Dépendances de l'API -----------------------------------------
# better-sqlite3 est natif : python3/make/g++ servent de secours si aucun
# binaire pré-compilé ne correspond à la plateforme.
FROM node:22-slim AS deps
WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 make g++ \
 && rm -rf /var/lib/apt/lists/*
COPY server/package.json server/package-lock.json ./
RUN npm ci --omit=dev

# --- 3. Image finale --------------------------------------------------
FROM node:22-slim
ENV NODE_ENV=production \
    PORT=3001 \
    ATELIER_CONTENT=/app/content \
    ATELIER_DB=/app/server/data/data.db

WORKDIR /app/server
COPY --from=deps /build/node_modules ./node_modules
COPY server/ ./
COPY content/ /app/content/
COPY --from=web /build/dist /app/web/dist

# La base vit dans un volume : le dossier doit exister et appartenir à node.
RUN rm -f data.db data.db-shm data.db-wal \
 && mkdir -p /app/server/data \
 && chown -R node:node /app

USER node
EXPOSE 3001
CMD ["node", "index.js"]
