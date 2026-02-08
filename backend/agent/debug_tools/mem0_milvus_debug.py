from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from mem0 import Memory
from pymilvus import MilvusClient


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def build_mem0_config() -> Dict[str, Any]:
    """
    Must match your production mem0 config logic.
    IMPORTANT: do not pass token/db_name when they are None.
    """
    url = _env("MEM0_MILVUS_URL", "http://127.0.0.1:19530")
    collection = _env("MEM0_MILVUS_COLLECTION", "simu_agent_mem0")
    dims = int(_env("MEM0_EMBED_DIMS", "1536") or "1536")
    version = _env("MEM0_VERSION", "v1.1")

    token = _env("MEM0_MILVUS_TOKEN", None)
    db_name = _env("MEM0_MILVUS_DB_NAME", None)

    milvus_cfg: Dict[str, Any] = {
        "collection_name": collection,
        "embedding_model_dims": dims,
        "url": url,
    }
    if token:
        milvus_cfg["token"] = token
    if db_name:
        milvus_cfg["db_name"] = db_name

    return {
        "vector_store": {
            "provider": "milvus",
            "config": milvus_cfg,
        },
        "version": version,
    }


def get_mem() -> Memory:
    cfg = build_mem0_config()
    return Memory.from_config(cfg)


def get_milvus_client() -> MilvusClient:
    url = _env("MEM0_MILVUS_URL", "http://127.0.0.1:19530")
    token = _env("MEM0_MILVUS_TOKEN", None)
    db_name = _env("MEM0_MILVUS_DB_NAME", None)

    kwargs: Dict[str, Any] = {"uri": url}
    if token:
        kwargs["token"] = token
    if db_name:
        kwargs["db_name"] = db_name

    return MilvusClient(**kwargs)


def get_collection_name() -> str:
    return _env("MEM0_MILVUS_COLLECTION", "simu_agent_mem0") or "simu_agent_mem0"


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# ---------------------------
# Milvus schema helpers
# ---------------------------

def collection_exists(client: MilvusClient, name: str) -> bool:
    try:
        return name in client.list_collections()
    except Exception:
        return False


def describe_collection(client: MilvusClient, name: str) -> Dict[str, Any]:
    return client.describe_collection(collection_name=name)


def _get_fields(desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields = desc.get("fields") or desc.get("schema", {}).get("fields") or []
    return [f for f in fields if isinstance(f, dict)]


def get_field_names(desc: Dict[str, Any]) -> List[str]:
    return [f.get("name") for f in _get_fields(desc) if f.get("name")]


def get_primary_field_name(desc: Dict[str, Any]) -> Optional[str]:
    for f in _get_fields(desc):
        if f.get("is_primary_key"):
            return f.get("name")
    return None


def get_pk_expr_for_all(desc: Dict[str, Any]) -> Optional[str]:
    """
    Build a "match all" expr that works on older Milvus versions that don't support 'true'.
    We try:
      - pk >= 0 for int pk
      - pk != "" for string pk
    """
    pk = get_primary_field_name(desc)
    if not pk:
        return None

    # Try infer pk dtype (keys differ by version)
    fpk = next((f for f in _get_fields(desc) if f.get("name") == pk), None)
    dtype = None
    if fpk:
        dtype = fpk.get("data_type") or fpk.get("type")  # best-effort

    # If dtype is int-like:
    if str(dtype).lower() in ("int64", "int32", "int", "long"):
        return f"{pk} >= 0"

    # Otherwise assume string-like:
    return f'{pk} != ""'


def query_rows(
    client: MilvusClient,
    name: str,
    expr: Optional[str],
    output_fields: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Milvus query filter compatibility:
    - Some versions reject 'true'
    - Some allow empty filter by omitting 'filter' param
    """
    kwargs: Dict[str, Any] = {
        "collection_name": name,
        "output_fields": output_fields,
        "limit": limit,
    }
    if expr:
        kwargs["filter"] = expr
    return client.query(**kwargs)


def delete_rows(
    client: MilvusClient,
    name: str,
    expr: Optional[str],
) -> Dict[str, Any]:
    """
    Delete compatibility:
    - Some versions reject 'true'
    - If expr is None/empty, we error explicitly (avoid accidental wipe)
    """
    if not expr:
        raise ValueError("delete_rows requires a non-empty expr")
    return client.delete(collection_name=name, filter=expr)


# ---------------------------
# Commands
# ---------------------------

def cmd_info(args: argparse.Namespace) -> None:
    client = get_milvus_client()
    name = get_collection_name()
    print("[env] MEM0_MILVUS_URL =", _env("MEM0_MILVUS_URL"))
    print("[env] MEM0_MILVUS_COLLECTION =", name)
    print("[env] MEM0_EMBED_DIMS =", _env("MEM0_EMBED_DIMS"))
    print("[milvus] list_collections =", client.list_collections())

    if not collection_exists(client, name):
        print(f"[milvus] collection '{name}' does NOT exist yet.")
        return

    desc = describe_collection(client, name)
    print("[milvus] describe_collection:\n", safe_json(desc))
    pk = get_primary_field_name(desc)
    print("[milvus] primary_field =", pk)


def cmd_mem0_add(args: argparse.Namespace) -> None:
    mem = get_mem()
    metadata = json.loads(args.metadata) if args.metadata else {}
    resp = mem.add(args.text, user_id=args.user_id, metadata=metadata)
    print("[mem0] add resp:\n", safe_json(resp))


def cmd_mem0_search(args: argparse.Namespace) -> None:
    mem = get_mem()
    resp = mem.search(query=args.query, user_id=args.user_id, limit=args.k)
    print("[mem0] search resp:\n", safe_json(resp))


def cmd_milvus_list(args: argparse.Namespace) -> None:
    client = get_milvus_client()
    name = get_collection_name()
    if not collection_exists(client, name):
        print(f"[milvus] collection '{name}' does NOT exist.")
        return

    desc = describe_collection(client, name)
    field_names = get_field_names(desc)

    preferred = [x for x in ["id", "memory", "user_id", "metadata", "created_at", "updated_at"] if x in field_names]
    output_fields = preferred if preferred else field_names[: min(len(field_names), 8)]

    expr: Optional[str] = args.expr if args.expr else None
    if not expr and args.user_id:
        if "user_id" in field_names:
            expr = f'user_id == "{args.user_id}"'
        else:
            expr = None  # can't filter by user_id field

    # If still no expr, use pk-based match-all
    if not expr:
        expr = get_pk_expr_for_all(desc)

    rows = query_rows(client, name, expr=expr, output_fields=output_fields, limit=args.limit)
    print(f"[milvus] query expr={expr} output_fields={output_fields} limit={args.limit}")
    print(safe_json(rows))


def cmd_milvus_delete(args: argparse.Namespace) -> None:
    client = get_milvus_client()
    name = get_collection_name()
    if not collection_exists(client, name):
        print(f"[milvus] collection '{name}' does NOT exist.")
        return

    desc = describe_collection(client, name)
    pk = get_primary_field_name(desc)

    expr: Optional[str] = None
    if args.id:
        if not pk:
            print("[milvus] cannot delete by id: primary key field not detected from schema")
            sys.exit(2)
        expr = f'{pk} in ["{args.id}"]'
    elif args.expr:
        expr = args.expr
    else:
        print("[milvus] delete requires --id or --expr")
        sys.exit(2)

    resp = delete_rows(client, name, expr=expr)
    print(f"[milvus] delete expr={expr}\n", safe_json(resp))


def cmd_milvus_clear(args: argparse.Namespace) -> None:
    client = get_milvus_client()
    name = get_collection_name()
    if not collection_exists(client, name):
        print(f"[milvus] collection '{name}' does NOT exist.")
        return

    if args.drop:
        resp = client.drop_collection(collection_name=name)
        print("[milvus] drop_collection resp:\n", safe_json(resp))
        print("[milvus] dropped.")
        return

    # delete all using pk-based expr
    desc = describe_collection(client, name)
    expr = get_pk_expr_for_all(desc)
    if not expr:
        raise RuntimeError("Cannot derive a safe match-all expr (no primary key). Use --drop instead.")
    resp = delete_rows(client, name, expr=expr)
    print(f"[milvus] delete all expr={expr}\n", safe_json(resp))


def cmd_update(args: argparse.Namespace) -> None:
    # delete first
    del_args = argparse.Namespace(id=args.id, expr=args.expr)
    cmd_milvus_delete(del_args)

    # then add
    add_args = argparse.Namespace(text=args.text, user_id=args.user_id, metadata=args.metadata)
    cmd_mem0_add(add_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mem0 + Milvus debug utility (CRUD / list / clear)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("info", help="Print env + list collections + describe target collection")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("add", help="mem0 add(text) -> Milvus")
    sp.add_argument("--user-id", required=True)
    sp.add_argument("--text", required=True)
    sp.add_argument("--metadata", default="", help='JSON string, e.g. \'{"action":"accept","is_correct":true}\'')
    sp.set_defaults(func=cmd_mem0_add)

    sp = sub.add_parser("search", help="mem0 semantic search(query) -> Milvus")
    sp.add_argument("--user-id", required=True)
    sp.add_argument("--query", required=True)
    sp.add_argument("-k", type=int, default=5)
    sp.set_defaults(func=cmd_mem0_search)

    sp = sub.add_parser("list", help="Milvus query rows (no embedding).")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--expr", default="", help='Milvus filter expr, e.g. \'user_id == "u1"\'')
    sp.add_argument("--user-id", default="", help="Shortcut: filter by user_id if field exists")
    sp.set_defaults(func=cmd_milvus_list)

    sp = sub.add_parser("delete", help="Milvus delete by id or expr.")
    sp.add_argument("--id", default="", help="Primary key value to delete")
    sp.add_argument("--expr", default="", help='Milvus filter expr')
    sp.set_defaults(func=cmd_milvus_delete)

    sp = sub.add_parser("update", help="Update = delete (milvus) + add (mem0)")
    sp.add_argument("--id", default="", help="Primary key value to delete first")
    sp.add_argument("--expr", default="", help="Milvus filter expr to delete first")
    sp.add_argument("--user-id", required=True)
    sp.add_argument("--text", required=True)
    sp.add_argument("--metadata", default="", help="JSON string metadata for new record")
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("clear", help="Clear all rows or drop collection.")
    sp.add_argument("--drop", action="store_true", help="Drop collection entirely.")
    sp.set_defaults(func=cmd_milvus_clear)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


'''
python .\mem0_milvus_debug.py info
python .\mem0_milvus_debug.py list --limit 20

python .\mem0_milvus_debug.py add --user-id u_debug --text "global_batch_size sweep find OOM boundary" --metadata "{}"
python .\mem0_milvus_debug.py list --user-id u_debug --limit 20
python .\mem0_milvus_debug.py search --user-id u_debug --query "batch size sweep boundary" -k 5

python .\mem0_milvus_debug.py delete --expr "user_id == 'u_debug'"
python .\mem0_milvus_debug.py clear
python .\mem0_milvus_debug.py clear --drop

default_user
python .\mem0_milvus_debug.py search --user-id default_user --query "batch size sweep boundary" -k 5
python .\mem0_milvus_debug.py  list --user-id default_user --limit 20
'''