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

# gosu : l'entrypoint démarre en root le temps d'ajuster le volume, puis
# bascule sur l'utilisateur node sans perdre les signaux (pas de PID 1 parasite).
RUN apt-get update \
 && apt-get install -y --no-install-recommends gosu \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/server
COPY --from=deps /build/node_modules ./node_modules
COPY server/ ./
COPY content/ /app/content/
COPY --from=web /build/dist /app/web/dist

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# La base vit dans un volume : le dossier doit exister et appartenir à node.
# Le sed neutralise les fins de ligne CRLF si le dépôt a été cloné sous Windows.
RUN rm -f data.db data.db-shm data.db-wal \
 && mkdir -p /app/server/data \
 && chown -R node:node /app \
 && sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
 && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 3001
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "index.js"]
