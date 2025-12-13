FROM python:3.12-slim-bookworm
LABEL authors="fastium"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# 1. Installation des outils système (git, docker, kubectl, yq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*

# 2. Installation de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Installation des outils externes (kubectl, yq)
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

RUN curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/bin/yq \
    && chmod +x /usr/bin/yq

WORKDIR /app

# --- PARTIE AJOUTÉE POUR PRÉ-INSTALLER LES DÉPENDANCES ---

# 4. On copie UNIQUEMENT les fichiers de requirements
COPY requirements-freeze.txt ./

# 5. On installe les dépendances avec uv
# On combine les deux installations pour avoir une seule image prête à tout faire.
# L'option --system installe directement dans le python global de l'image.
RUN uv pip install -r requirements-freeze.txt

# 6. (Optionnel) Nettoyage du cache uv pour alléger l'image
RUN uv cache clean

CMD ["bash"]