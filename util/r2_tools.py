import argparse
import logging
import os
import sys
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()


def build_s3_client() -> boto3.client:
	"""Builds an S3-compatible client for Cloudflare R2 using environment variables."""
	r2_account_id = os.getenv("CF_ID")
	r2_access_key_id = os.getenv("R2_ID")
	r2_secret_access_key = os.getenv("R2_KEY")
	custom_endpoint = os.getenv("R2_ENDPOINT")

	if not r2_account_id and not custom_endpoint:
		raise RuntimeError("R2_ACCOUNT_ID or R2_ENDPOINT must be set in the environment.")
	if not r2_access_key_id or not r2_secret_access_key:
		raise RuntimeError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set in the environment.")

	endpoint_url = custom_endpoint or f"https://{r2_account_id}.r2.cloudflarestorage.com"

	return boto3.client(
		"s3",
		aws_access_key_id=r2_access_key_id,
		aws_secret_access_key=r2_secret_access_key,
		endpoint_url=endpoint_url,
		region_name="auto",
		config=Config(signature_version="s3v4"),
	)


def normalize_prefix(prefix: str) -> str:
	"""Ensures a clean prefix without leading slash and with a trailing slash."""
	if prefix.startswith("/"):
		prefix = prefix[1:]
	if prefix and not prefix.endswith("/"):
		prefix = prefix + "/"
	return prefix


def count_files(bucket: str, prefix: str, recursive: bool = True) -> int:
	"""Counts the number of files under a given prefix.

	If recursive is False, counts only immediate files (not in subfolders) using Delimiter='/' for speed.
	If recursive is True, counts all files recursively.
	"""
	s3 = build_s3_client()
	prefix = normalize_prefix(prefix)
	paginator = s3.get_paginator("list_objects_v2")
	params = {"Bucket": bucket, "Prefix": prefix}
	if not recursive:
		params["Delimiter"] = "/"
		total = 0
		for page in paginator.paginate(**params):
			for obj in page.get("Contents", []):
				key = obj.get("Key", "")
				# Skip folder marker objects
				if key.endswith("/") and obj.get("Size", 0) == 0:
					continue
				total += 1
		return total
	else:
		total = 0
		for page in paginator.paginate(**params):
			for obj in page.get("Contents", []):
				key = obj.get("Key", "")
				if key.endswith("/") and obj.get("Size", 0) == 0:
					continue
				total += 1
		return total


def count_folders(bucket: str, prefix: str, recursive: bool = False) -> int:
	"""Counts folders under a prefix using Delimiter='/' for speed.

	If recursive is False, counts immediate subfolders only.
	If recursive is True, traverses subfolders breadth-first and counts unique folders.
	"""
	s3 = build_s3_client()
	prefix = normalize_prefix(prefix)
	paginator = s3.get_paginator("list_objects_v2")

	def list_immediate_subfolders(parent_prefix: str):
		params = {"Bucket": bucket, "Prefix": parent_prefix, "Delimiter": "/"}
		for page in paginator.paginate(**params):
			for cp in page.get("CommonPrefixes", []) or []:
				yield cp.get("Prefix")

	if not recursive:
		count = 0
		for _ in list_immediate_subfolders(prefix):
			count += 1
		return count
	else:
		seen = set()
		queue = [prefix]
		count = 0
		while queue:
			current = queue.pop(0)
			for sub in list_immediate_subfolders(current):
				if sub not in seen:
					seen.add(sub)
					count += 1
					queue.append(sub)
		return count


def list_folders(bucket: str, prefix: str, recursive: bool = False):
	"""Lists folders under a prefix using Delimiter='/' for speed.

	If recursive is False, lists immediate subfolders only.
	If recursive is True, traverses subfolders breadth-first and lists unique folders.
	"""
	s3 = build_s3_client()
	prefix = normalize_prefix(prefix)
	paginator = s3.get_paginator("list_objects_v2")

	def list_immediate_subfolders(parent_prefix: str):
		params = {"Bucket": bucket, "Prefix": parent_prefix, "Delimiter": "/"}
		for page in paginator.paginate(**params):
			for cp in page.get("CommonPrefixes", []) or []:
				yield cp.get("Prefix")

	if not recursive:
		return list(list_immediate_subfolders(prefix))
	else:
		seen = set()
		queue = [prefix]
		folders = []
		while queue:
			current = queue.pop(0)
			for sub in list_immediate_subfolders(current):
				if sub not in seen:
					seen.add(sub)
					folders.append(sub)
					queue.append(sub)
		return folders


def move_prefix(bucket: str, from_prefix: str, to_prefix: str, dry_run: bool = False, limit: Optional[int] = None, workers: int = 16, delete_batch_size: int = 1000) -> int:
	"""Moves all objects from one prefix to another by copy+delete.

	Performs concurrent copies and batches deletes for speed. Returns the number of objects moved.
	If limit is provided, moves at most that many objects.
	"""
	s3 = build_s3_client()
	from_prefix = normalize_prefix(from_prefix)
	to_prefix = normalize_prefix(to_prefix)

	if from_prefix == to_prefix:
		raise ValueError("from_prefix and to_prefix must be different.")
	if to_prefix.startswith(from_prefix):
		raise ValueError("to_prefix cannot be a child of from_prefix.")

	paginator = s3.get_paginator("list_objects_v2")

	def iter_source_objects():
		count = 0
		for page in paginator.paginate(Bucket=bucket, Prefix=from_prefix):
			for obj in page.get("Contents", []) or []:
				key = obj.get("Key", "")
				if not key or key.endswith("/"):
					continue
				yield key
				count += 1
				if limit is not None and count >= limit:
					return

	# Dry run: just log and count
	if dry_run:
		moved = 0
		for key in iter_source_objects():
			relative = key[len(from_prefix):]
			if len(relative) == len(key):
				continue
			new_key = to_prefix + relative
			logging.info(f"Would move s3://{bucket}/{key} -> s3://{bucket}/{new_key}")
			moved += 1
		return moved

	def copy_one(src_key: str, dst_key: str) -> bool:
		try:
			s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dst_key)
			return True
		except ClientError as e:
			logging.error(f"Failed to copy {src_key} -> {dst_key}: {e}")
			return False

	moved = 0
	delete_buffer = []

	with ThreadPoolExecutor(max_workers=workers) as executor:
		future_to_key = {}
		for key in iter_source_objects():
			relative = key[len(from_prefix):]
			if len(relative) == len(key):
				continue
			new_key = to_prefix + relative
			fut = executor.submit(copy_one, key, new_key)
			future_to_key[fut] = key

		for fut in as_completed(future_to_key):
			key = future_to_key[fut]
			ok = fut.result()
			if ok:
				delete_buffer.append({"Key": key})
				moved += 1
				if len(delete_buffer) >= min(delete_batch_size, 1000):
					try:
						s3.delete_objects(Bucket=bucket, Delete={"Objects": delete_buffer, "Quiet": True})
						delete_buffer.clear()
					except ClientError as e:
						logging.error(f"Batch delete failed: {e}")

	# Final flush
	if delete_buffer:
		try:
			s3.delete_objects(Bucket=bucket, Delete={"Objects": delete_buffer, "Quiet": True})
		except ClientError as e:
			logging.error(f"Final batch delete failed: {e}")

	return moved


def main(argv: Optional[list] = None) -> int:
	parser = argparse.ArgumentParser(description="Cloudflare R2 tools: count and move objects between prefixes.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	parser.add_argument("--bucket", required=True, help="R2 bucket name")
	parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

	count_parser = subparsers.add_parser("count", help="Count items under a prefix")
	count_parser.add_argument("--prefix", required=True, help="Prefix to count, e.g., tasks/processing_todo")
	count_parser.add_argument("--kind", choices=["folders", "files"], default="folders", help="Count folders (default) or files")
	count_parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders when counting")

	list_parser = subparsers.add_parser("list-folders", help="List folders under a prefix")
	list_parser.add_argument("--prefix", required=True, help="Prefix to list, e.g., processed/en")
	list_parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders when listing")

	move_parser = subparsers.add_parser("move", help="Move objects from one prefix to another")
	move_parser.add_argument("--from", dest="from_prefix", required=True, help="Source prefix, e.g., tasks/processing_in_progress")
	move_parser.add_argument("--to", dest="to_prefix", required=True, help="Destination prefix, e.g., tasks/processing_todo")
	move_parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without performing changes")
	move_parser.add_argument("--limit", type=int, default=None, help="Max number of objects to move")
	move_parser.add_argument("--workers", type=int, default=16, help="Number of concurrent copy workers")
	move_parser.add_argument("--delete-batch-size", type=int, default=1000, help="Number of deletes to batch per request (max 1000)")

	args = parser.parse_args(argv)
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)

	if args.command == "count":
		if args.kind == "folders":
			total = count_folders(args.bucket, args.prefix, recursive=args.recursive)
		else:
			total = count_files(args.bucket, args.prefix, recursive=args.recursive)
		print(total)
		return 0
	elif args.command == "list-folders":
		folders = list_folders(args.bucket, args.prefix, recursive=args.recursive)
		for f in folders:
			print(f)
		return 0
	elif args.command == "move":
		moved = move_prefix(args.bucket, args.from_prefix, args.to_prefix, dry_run=args.dry_run, limit=args.limit, workers=args.workers, delete_batch_size=args.delete_batch_size)
		print(moved)
		return 0
	else:
		parser.print_help()
		return 1


if __name__ == "__main__":
	sys.exit(main()) 