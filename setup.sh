#!bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git@main#egg=clip
echo "Setup complete!"