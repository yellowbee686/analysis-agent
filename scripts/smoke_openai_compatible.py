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
from local_file_agent.llm import (
    build_openai_clients,
    extract_response_text,
    get_model_params,
    list_models,
    select_endpoint,
)


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
        "LOCAL_AGENT_MODEL_TYPE",
        "deepseek-ai/deepseek-v3.2",
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

    if args.no_azure:
        use_azure = False
    else:
        use_azure = endpoint.use_azure

    client_mode = "auto"
    if use_azure is True:
        client_mode = "AzureOpenAI"
    elif use_azure is False:
        client_mode = "OpenAI"

    logger.info("Using %s client", client_mode)

    client, _ = build_openai_clients(endpoint, use_azure=use_azure)

    model_params = get_model_params(model_name)
    if model_params:
        logger.info("Using model params for '%s': %s", model_name, model_params)

    logger.info("Sending test request to model '%s'...", model_name)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            **model_params,
        )
        logger.info("Response received successfully!")
        text = extract_response_text(response)
        print(f"Model response: {text}")
        if not text:
            logger.warning(
                "Empty response text. The model may return reasoning_content or "
                "multi-part content not captured by the standard field."
            )
    except Exception as e:
        logger.error("Request failed: %s", e)
        raise


if __name__ == "__main__":
    main()
