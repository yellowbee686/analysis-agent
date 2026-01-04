from __future__ import annotations

import os

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from local_file_agent.llm import build_openai_clients, select_endpoint

def main() -> None:
    load_dotenv()
    model_name = os.environ.get(
        "LOCAL_AGENT_MODEL_TYPE", "gemini-3-pro-preview-new"
    )
    endpoint = select_endpoint(model_name)
    if not endpoint:
        raise SystemExit(f"No endpoint configured for {model_name}")

    client, _ = build_openai_clients(endpoint)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=8,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
