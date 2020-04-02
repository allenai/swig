#!/bin/bash

rsync -avz \
  --exclude .idea \
  --exclude __pycache__/ \
  --exclude runs/ \
  --exclude .git/ \
  --exclude .DS_Store \
  --exclude venv/ \
   ../clean_gsr sarahp@172.16.2.125:/home/sarahp