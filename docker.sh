#!/bin/bash
docker stop $(docker ps -q)
docker build -t meteo:latest .
docker run -d -p 5000:5000 meteo:latest