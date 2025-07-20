#!/bin/bash

host="$1"
port="$2"

until pg_isready -h "$host" -p "$port"; do
  echo "Waiting for Postgres at $host:$port..."
  sleep 2
done

echo "Postgres is ready!"