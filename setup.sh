#!/bin/bash
docker build . -t vittu
docker run -p 5000:5000 -v "$PWD":/src --name vittu vittu
