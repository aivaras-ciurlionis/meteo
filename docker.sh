#!/bin/bash
docker stop $(docker ps -q)
docker build -t meteo:latest .
docker run -d -p 5000:8080 meteo:latest