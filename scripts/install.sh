#!/bin/bash
set -e

# install uv
pip install -U uv

# create venv
uv venv .venv --python=3.12 --seed 

# activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

uv sync
