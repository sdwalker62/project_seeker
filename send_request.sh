#!/bin/bash

echo -e "\n"

curl -i \
    -H "Content-type: application/json" \
    -H "Accept: application/json" \
    http://127.0.0.1:8000/$1?prompt=$2

echo -e "\n"
