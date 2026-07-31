"""Verify metadata.pkl integrity against the FAISS index."""

import os
import pickle
import sys

import faiss

# ── Paths ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "database", "metadata.pkl")
INDEX_PATH = os.path.join(BASE_DIR, "database", "faiss.index")

# ── Verification ─────────────────────────────────────────────────────

ok = True


def fail(msg):
    global ok
    ok = False
    print(msg)


# 1. File exists
print("=" * 56)
print("metadata.pkl Integrity Report")
print("=" * 56)

print()
print("1. metadata.pkl exists")
if os.path.exists(META_PATH):
    size = os.path.getsize(META_PATH)
    print(f"   YES  ({size} bytes)")
else:
    fail("   NO  (file not found at database/metadata.pkl)")
    print()
    print("=" * 56)
    print("metadata.pkl verification FAILED")
    print("=" * 56)
    sys.exit(1)

# 2. Load metadata
print()
print("2. Load metadata.pkl")
try:
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    n_meta = len(metadata) if isinstance(metadata, (list, tuple)) else 1
    print(f"   Loaded: {n_meta} entries")
except Exception as e:
    fail(f"   FAILED: {e}")
    print()
    print("=" * 56)
    print("metadata.pkl verification FAILED")
    print("=" * 56)
    sys.exit(1)

# 3. Types and samples
print()
print("3. Type information")
print(f"   Metadata object type  : {type(metadata).__name__}")
if isinstance(metadata, list) and metadata:
    print(f"   First entry type     : {type(metadata[0]).__name__}")
    if isinstance(metadata[0], dict):
        print(f"   First entry keys     : {list(metadata[0].keys())}")

print()
print("4. Sample entries")
n_meta = len(metadata) if isinstance(metadata, list) else 0
print(f"   Total metadata entries: {n_meta}")
if isinstance(metadata, list):
    if metadata:
        print("   First metadata entry:")
        entry = metadata[0]
        if isinstance(entry, dict):
            for k, v in entry.items():
                print(f"      {k}: {str(v)[:160]}")
        else:
            print(f"      (not a dict) {entry}")
    if len(metadata) > 1:
        print("   Last metadata entry:")
        entry = metadata[-1]
        if isinstance(entry, dict):
            for k, v in entry.items():
                print(f"      {k}: {str(v)[:160]}")
        else:
            print(f"      (not a dict) {entry}")

# 5. FAISS index
print()
print("5. FAISS index")
try:
    index = faiss.read_index(INDEX_PATH)
    n_vec = index.ntotal
    print(f"   FAISS vectors        : {n_vec}")
except Exception as e:
    fail(f"   FAILED: {e}")
    n_vec = -1

# 6. Count comparison
print()
print("6. Metadata count vs FAISS vector count")
if n_vec >= 0 and n_meta >= 0:
    print(f"   Metadata entries     : {n_meta}")
    print(f"   FAISS vectors        : {n_vec}")
    if n_meta == n_vec:
        print("   COUNT MATCH          : PASS")
    else:
        fail(f"   COUNT MATCH          : FAIL  (difference: {abs(n_meta - n_vec)})")

# 7. Integrity
print()
print("7. Entry integrity")
if isinstance(metadata, list):
    required = {"pdf_name", "page", "chunk", "text"}
    bad_type = missing = empty_text = 0
    for entry in metadata:
        if not isinstance(entry, dict):
            bad_type += 1
            continue
        miss = required - set(entry.keys())
        if miss:
            missing += 1
        if not entry.get("text"):
            empty_text += 1
    for label, count in [("Non-dict entries", bad_type),
                          ("Missing fields  ", missing),
                          ("Empty texts     ", empty_text)]:
        print(f"   {label}: {count}" + (" (OK)" if count == 0 else " (FAIL)"))

# ── Summary ─────────────────────────────────────────────────────────

print()
print("=" * 56)
if ok:
    print(" metadata.pkl verification PASSED")
    print(" All checks OK")
else:
    print(" metadata.pkl verification FAILED")
    print(" Review the issues above and run rebuild_index() if needed")
print("=" * 56)

sys.exit(0 if ok else 1)
