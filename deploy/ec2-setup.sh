#!/bin/bash
# EC2 Setup Script for Personal Memory System
# Run this on a fresh Ubuntu 22.04 g4dn.xlarge instance

set -e

echo "=== Updating system ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Python 3.11 and dependencies ==="
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx certbot python3-certbot-nginx

echo "=== Installing NVIDIA drivers ==="
sudo apt install -y nvidia-driver-535 nvidia-utils-535

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Starting Ollama service ==="
sudo systemctl enable ollama
sudo systemctl start ollama

echo "=== Waiting for Ollama to be ready ==="
sleep 10

echo "=== Pulling models (this will take a while) ==="
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull phi3:14b

echo "=== Cloning repository ==="
cd ~
git clone https://github.com/YOUR_USERNAME/memory.git
cd memory

echo "=== Creating virtual environment ==="
python3.11 -m venv .venv
source .venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -e ".[web]"

echo "=== Creating data directory ==="
mkdir -p ~/memory/data

echo "=== Installing systemd service ==="
sudo cp deploy/memory.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable memory

echo "=== Installing nginx config ==="
sudo cp deploy/nginx.conf /etc/nginx/sites-available/memory
sudo ln -sf /etc/nginx/sites-available/memory /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo "=== Setup complete! ==="
echo "Next steps:"
echo "1. Run: python scripts/setup_auth.py (to create admin user)"
echo "2. Run: sudo certbot --nginx -d YOUR_DOMAIN"
echo "3. Run: sudo systemctl start memory"
