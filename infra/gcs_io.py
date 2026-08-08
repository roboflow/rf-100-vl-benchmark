#!/usr/bin/env python3
"""Small, credential-safe GCS helpers for the RunPod wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys
from urllib.parse import urlparse


def parse_uri(uri: str, *, require_object: bool = True) -> tuple[str, str]:
    parsed = urlparse(uri)
    prefix = parsed.path.strip("/")
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    if require_object and not prefix:
        raise ValueError("A GCS object or prefix is required.")
    if prefix and any(part in ("", ".", "..") for part in PurePosixPath(prefix).parts):
        raise ValueError("Unsafe GCS object path.")
    return parsed.netloc, prefix


def client():
    from google.cloud import storage

    return storage.Client()


def upload(root_uri: str, source: Path, relative_path: str) -> str:
    bucket, prefix = parse_uri(root_uri)
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("Unsafe relative GCS path.")
    object_name = f"{prefix}/{relative.as_posix()}"
    client().bucket(bucket).blob(object_name).upload_from_filename(str(source))
    return f"gs://{bucket}/{object_name}"


def download(uri: str, destination: Path) -> None:
    bucket, object_name = parse_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.gcs.tmp")
    client().bucket(bucket).blob(object_name).download_to_filename(str(temporary))
    temporary.replace(destination)


def download_prefix(uri: str, destination: Path) -> int:
    bucket, prefix = parse_uri(uri)
    object_prefix = prefix.rstrip("/") + "/"
    restored = 0
    storage_client = client()
    for blob in storage_client.list_blobs(bucket, prefix=object_prefix):
        suffix = blob.name[len(object_prefix) :]
        if not suffix:
            continue
        suffix_path = PurePosixPath(suffix)
        if suffix_path.is_absolute() or ".." in suffix_path.parts:
            raise ValueError(f"Unsafe object below dataset prefix: {blob.name}")
        local_path = destination.joinpath(*suffix_path.parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(f".{local_path.name}.gcs.tmp")
        blob.download_to_filename(str(temporary))
        temporary.replace(local_path)
        restored += 1
    return restored


def exists(uri: str) -> bool:
    bucket, object_name = parse_uri(uri)
    storage_client = client()
    return storage_client.bucket(bucket).blob(object_name).exists(client=storage_client)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--root-uri", required=True)
    upload_parser.add_argument("--source", type=Path, required=True)
    upload_parser.add_argument("--relative-path", required=True)

    optional_parser = subparsers.add_parser("upload-if-possible")
    optional_parser.add_argument("--root-uri", default="")
    optional_parser.add_argument("--source", type=Path, required=True)
    optional_parser.add_argument("--relative-path", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--uri", required=True)
    download_parser.add_argument("--destination", type=Path, required=True)

    prefix_parser = subparsers.add_parser("download-prefix")
    prefix_parser.add_argument("--uri", required=True)
    prefix_parser.add_argument("--destination", type=Path, required=True)

    exists_parser = subparsers.add_parser("exists")
    exists_parser.add_argument("--uri", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "upload-if-possible":
        if not args.root_uri or not args.source.is_file():
            return 0
        try:
            upload(args.root_uri, args.source, args.relative_path)
        except Exception as error:
            print(f"[gcs] final log upload failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 0
    if args.command == "upload":
        print(upload(args.root_uri, args.source, args.relative_path))
    elif args.command == "download":
        download(args.uri, args.destination)
    elif args.command == "download-prefix":
        print(download_prefix(args.uri, args.destination))
    elif args.command == "exists":
        return 0 if exists(args.uri) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
