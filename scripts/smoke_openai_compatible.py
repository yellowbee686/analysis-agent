from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from local_file_agent.llm import build_openai_clients, list_models, select_endpoint


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenAI compatible API")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-azure", action="store_true", help="Use standard OpenAI client instead of Azure")
    parser.add_argument("--model", type=str, help="Model name to test")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    load_dotenv()

    # List available models
    available_models = list_models()
    logger.info("Available models: %s", available_models)

    model_name = args.model or os.environ.get(
        "LOCAL_AGENT_MODEL_TYPE", "gemini-3-pro-preview-new"
    )
    logger.info("Testing model: %s", model_name)

    endpoint = select_endpoint(model_name)
    if not endpoint:
        raise SystemExit(f"No endpoint configured for {model_name}")

    logger.info(
        "Endpoint selected: base_url=%s, api_key=%s...%s",
        endpoint.base_url,
        endpoint.api_key[:5] if endpoint.api_key else "None",
        endpoint.api_key[-3:] if endpoint.api_key and len(endpoint.api_key) > 8 else "",
    )

    use_azure = not args.no_azure
    logger.info("Using %s client", "AzureOpenAI" if use_azure else "OpenAI")

    client, _ = build_openai_clients(endpoint, use_azure=use_azure)

    logger.info("Sending test request to model '%s'...", model_name)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
        logger.info("Response received successfully!")
        print(f"Model response: {response.choices[0].message.content}")
    except Exception as e:
        logger.error("Request failed: %s", e)
        raise


if __name__ == "__main__":
    main()
