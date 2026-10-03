#!/usr/bin/env python3
"""
Migrate omega embeddings to a different backend / dimension.

Usage:
    # Switch to OpenAI (keeps 384 dim — no schema change needed):
    OMEGA_EMBEDDING_BACKEND=openai OPENAI_API_KEY=sk-... python3 scripts/migrate_embeddings.py

    # Switch to Voyage (new dim — drops and recreates vec table):
    OMEGA_EMBEDDING_BACKEND=voyage VOYAGE_API_KEY=... \\
    OMEGA_EMBEDDING_DIM=1024 OMEGA_VOYAGE_MODEL=voyage-3 \\
    python3 scripts/migrate_embeddings.py

    # Dry-run (no writes):
    python3 scripts/migrate_embeddings.py --dry-run
"""

import argparse
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

DB_PATH = Path(os.environ.get("OMEGA_DB_PATH", "~/.omega/omega.db")).expanduser()
BATCH_SIZE = 64


def _serialize_f32(values: list) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _get_new_dim() -> int:
    return int(os.environ.get("OMEGA_EMBEDDING_DIM", "384"))


def _embed_batch(texts: list) -> list:
    """Call the configured API backend. Raises on failure."""
    backend = os.environ.get("OMEGA_EMBEDDING_BACKEND", "").lower()
    dim = _get_new_dim()

    if backend == "openai":
        from openai import OpenAI
        model = os.environ.get("OMEGA_OPENAI_MODEL", "text-embedding-3-small")
        client = OpenAI()
        resp = client.embeddings.create(model=model, input=texts, dimensions=dim)
        return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    elif backend == "voyage":
        import voyageai
        model = os.environ.get("OMEGA_VOYAGE_MODEL", "voyage-3-lite")
        client = voyageai.Client()
        resp = client.embed(texts, model=model)
        return resp.embeddings
    else:
        raise ValueError(f"Unknown OMEGA_EMBEDDING_BACKEND: {backend!r}. Set to 'openai' or 'voyage'.")


def _current_vec_dim(conn: sqlite3.Connection) -> int | None:
    """Detect current vec table dimension from stored chunks."""
    row = conn.execute(
        "SELECT length(vectors) FROM memories_vec_vector_chunks00 LIMIT 1"
    ).fetchone()
    if not row:
        return None
    # Each chunk holds 1024 vectors; bytes / 1024 / 4 = floats per vector
    return row[0] // 1024 // 4


def main():
    parser = argparse.ArgumentParser(description="Migrate omega embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing")
    args = parser.parse_args()

    backend = os.environ.get("OMEGA_EMBEDDING_BACKEND", "").lower()
    if not backend:
        print("ERROR: Set OMEGA_EMBEDDING_BACKEND=openai or =voyage", file=sys.stderr)
        sys.exit(1)

    new_dim = _get_new_dim()
    print(f"Migration target: backend={backend}, dim={new_dim}")
    print(f"DB: {DB_PATH}")
    if args.dry_run:
        print("[DRY RUN] No writes will be made.")

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    current_dim = _current_vec_dim(conn)
    print(f"Current vec dimension: {current_dim}")

    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"Total memories: {total}")

    dim_changed = current_dim is not None and current_dim != new_dim

    if dim_changed:
        print(f"\nDimension change: {current_dim} → {new_dim}")
        print("This requires dropping and recreating the memories_vec table.")
        if not args.dry_run:
            confirm = input("Proceed? This is irreversible without a backup. [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                sys.exit(0)
            print("Dropping memories_vec...")
            conn.execute("DROP TABLE IF EXISTS memories_vec")
            conn.execute("DROP TABLE IF EXISTS memories_vec_info")
            conn.execute("DROP TABLE IF EXISTS memories_vec_chunks")
            conn.execute("DROP TABLE IF EXISTS memories_vec_rowids")
            conn.execute("DROP TABLE IF EXISTS memories_vec_vector_chunks00")
            conn.commit()
            # Recreate — requires sqlite-vec to be loaded
            try:
                conn.enable_load_extension(True)
                import glob
                for path in glob.glob("/opt/homebrew/lib/sqlite-vec*"):
                    try:
                        conn.load_extension(path.replace(".dylib", ""))
                        break
                    except Exception:
                        pass
            except Exception:
                pass
            conn.execute(
                f"CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[{new_dim}] distance_metric=cosine)"
            )
            conn.commit()
            print(f"memories_vec recreated with dim={new_dim}")
    else:
        print(f"\nSame dimension ({new_dim}) — updating embeddings in-place.")

    # Fetch all memories
    rows = conn.execute("SELECT id, content FROM memories ORDER BY id").fetchall()
    print(f"\nEmbedding {len(rows)} memories...")

    ok = 0
    failed = 0
    t0 = time.time()

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] or "" for r in batch]

        try:
            embeddings = _embed_batch(texts)
        except Exception as e:
            print(f"  [WARN] batch {i//BATCH_SIZE} failed: {e}")
            failed += len(batch)
            continue

        if not args.dry_run:
            for row_id, emb in zip(ids, embeddings):
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row_id,))
                conn.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (row_id, _serialize_f32(emb)),
                )
            conn.commit()

        ok += len(batch)
        elapsed = time.time() - t0
        rate = ok / elapsed
        remaining = (len(rows) - ok - failed) / rate if rate > 0 else 0
        print(f"  {ok}/{len(rows)} done ({rate:.1f}/s, ~{remaining:.0f}s remaining)", end="\r")

    print(f"\n\nDone. OK={ok} failed={failed} time={time.time()-t0:.1f}s")
    if args.dry_run:
        print("[DRY RUN] No changes written.")
    conn.close()


if __name__ == "__main__":
    main()
