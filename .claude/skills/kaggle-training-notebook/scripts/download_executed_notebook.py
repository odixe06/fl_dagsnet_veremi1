#!/usr/bin/env python
"""Download Kaggle's executed ``__notebook__.ipynb`` with existing CLI OAuth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="Full Kaggle kernel ref: owner/slug")
    parser.add_argument("--output", required=True, type=Path, help="Destination .ipynb path")
    parser.add_argument("--version", type=int, help="Optional kernel version number")
    return parser.parse_args()


def split_kernel_ref(value: str) -> tuple[str, str]:
    parts = value.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("kernel must be the full owner/slug reference")
    return parts[0], parts[1]


def fail_without_url(exc: requests.RequestException, kernel: str, version: int | None) -> None:
    """Raise a useful download error without leaking a signed URL."""
    status = exc.response.status_code if exc.response is not None else "unknown"
    version_text = f", version {version}" if version is not None else ""
    raise RuntimeError(
        f"Kaggle executed-notebook download failed for {kernel}{version_text} "
        f"(HTTP {status}; signed URL suppressed)"
    ) from None


def main() -> None:
    args = parse_args()
    owner, slug = split_kernel_ref(args.kernel)

    api = KaggleApi()
    api.authenticate()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = owner
    request.kernel_slug = slug
    request.file_path = "__notebook__.ipynb"
    if args.version is not None:
        request.version_number = args.version

    try:
        with api.build_kaggle_client() as client:
            redirect = client.kernels.kernels_api_client.download_kernel_output(request)
    except requests.RequestException as exc:
        fail_without_url(exc, args.kernel, args.version)

    try:
        response = requests.get(redirect.url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        fail_without_url(exc, args.kernel, args.version)

    # Reject a source-only or malformed response without exposing its contents.
    notebook = json.loads(response.content)
    output_count = sum(len(cell.get("outputs", [])) for cell in notebook.get("cells", []))
    if output_count == 0:
        raise RuntimeError("downloaded notebook has zero rendered outputs; destination not written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print(f"downloaded {args.output} ({args.output.stat().st_size} bytes, {output_count} outputs)")


if __name__ == "__main__":
    main()
