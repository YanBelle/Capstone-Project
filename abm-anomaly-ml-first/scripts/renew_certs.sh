#!/usr/bin/env bash
set -euo pipefail

cd /home/$USER/Capstone-Project

docker compose run --rm certbot renew --webroot --webroot-path=/var/www/certbot
docker compose exec nginx nginx -s reload