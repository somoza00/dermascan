# Dockerfile da DermaScan Web — build multi-stage, serve estático via nginx.

FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# VITE_API_URL é lido em build-time (import.meta.env) — o padrão já bate
# com o serviço "api" do docker-compose.yml quando publicado em
# localhost:8000 no host. Sobrescreva com --build-arg pra outros cenários.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build


FROM nginx:1.27-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
