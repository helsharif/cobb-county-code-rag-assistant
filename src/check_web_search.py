"""Small diagnostic for verifying fallback web search."""

from __future__ import annotations

import argparse

from src.tools import web_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the fallback web-search tool.")
    parser.add_argument("query", nargs="*", default=["Cobb County fire inspection requirements"])
    args = parser.parse_args()
    query = " ".join(args.query)
    print(web_search(query))


if __name__ == "__main__":
    main()
