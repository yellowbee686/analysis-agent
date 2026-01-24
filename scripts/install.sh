# install uv
pip install -U uv
# create venv
uv venv .venv --python=3.12 --seed 
source .venv/bin/activate
uv sync