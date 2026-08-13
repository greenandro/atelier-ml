#!/bin/sh
set -e

# Quand ./data n'existe pas encore sur l'hôte, Docker crée le point de montage
# en root:root : l'utilisateur node ne pourrait pas y ouvrir la base SQLite.
# On rectifie ici, puis on rend la main à node.
DATA_DIR=$(dirname "${ATELIER_DB:-/app/server/data/data.db}")
mkdir -p "$DATA_DIR"
chown -R node:node "$DATA_DIR" 2>/dev/null || true

exec gosu node "$@"
