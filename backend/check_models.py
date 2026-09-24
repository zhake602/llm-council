"""Check configured model ids against the OpenRouter catalog.

A wrong id does not crash the council: that model just fails and is silently
left out. Run this after changing COUNCIL_MODELS or CHAIRMAN_MODEL:

    uv run python -m backend.check_models

It also lists the newest models from each configured provider, so you can
see when an upgrade is available.
"""

from collections import defaultdict

import httpx

from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, REASONING_EFFORT

MODELS_URL = "https://openrouter.ai/api/v1/models"
NEWEST_PER_PROVIDER = 5


def main() -> None:
    response = httpx.get(MODELS_URL, timeout=30.0)
    response.raise_for_status()
    catalog = {m["id"]: m for m in response.json()["data"]}

    configured = list(dict.fromkeys(COUNCIL_MODELS + [CHAIRMAN_MODEL]))
    print(f"Reasoning effort: {REASONING_EFFORT or 'off'}\n")
    print("Configured models:")
    ok = True
    for model_id in configured:
        role = "chairman" if model_id == CHAIRMAN_MODEL else "council"
        if model_id == CHAIRMAN_MODEL and model_id in COUNCIL_MODELS:
            role = "council + chairman"
        model = catalog.get(model_id)
        if model is None:
            ok = False
            print(f"  MISSING   {model_id} ({role}) - not in the OpenRouter catalog")
            continue
        reasoning = "reasoning" in (model.get("supported_parameters") or [])
        print(f"  ok        {model_id} ({role}){'' if reasoning else ' - no reasoning support'}")

    by_provider = defaultdict(list)
    for model in catalog.values():
        by_provider[model["id"].split("/")[0]].append(model)

    print(f"\nNewest models per configured provider (top {NEWEST_PER_PROVIDER}):")
    for provider in dict.fromkeys(m.split("/")[0] for m in configured):
        print(f"  {provider}:")
        newest = sorted(by_provider[provider], key=lambda m: m.get("created", 0), reverse=True)
        for model in newest[:NEWEST_PER_PROVIDER]:
            marker = "*" if model["id"] in configured else " "
            print(f"    {marker} {model['id']}")

    if not ok:
        raise SystemExit("\nSome configured models are missing - fix COUNCIL_MODELS / CHAIRMAN_MODEL in .env")


if __name__ == "__main__":
    main()
