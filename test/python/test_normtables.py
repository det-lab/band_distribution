"""
Tests for python/normtables.py: register a table file, fetch it by name from a
local HTTP server standing in for Zenodo, reuse the cache with no network,
refuse a file whose hash does not match, and name the registered tables when
asked for an unknown one.  Needs pooch; no Fortran, no network.

Run from the repository root:  python test/python/test_normtables.py
"""

import functools
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
import normtables as nt

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


tmp = tempfile.mkdtemp()
served, cache = os.path.join(tmp, "served"), os.path.join(tmp, "cache")
os.makedirs(served)
table = os.path.join(served, "norm_NR_ep2.5-350_eq0.75-200.h5")
with open(table, "wb") as f:
    f.write(b"stand-in table bytes\n" * 100)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


handler = functools.partial(QuietHandler, directory=served)
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"
os.environ["BAND_TABLES_DIR"] = cache

registry = os.path.join(tmp, "registry.json")
with open(registry, "w") as f:
    json.dump({"tables": {}}, f)
try:
    nt.register([table], url_base=base, registry=registry)
    reg = nt.load_registry(registry)
    e = reg.get("NR_ep2.5-350_eq0.75-200", {})
    check("register: name from the file name, hash and URL recorded",
          e.get("file") == "norm_NR_ep2.5-350_eq0.75-200.h5" and e.get("sha256") == nt.sha256(table)
          and e.get("url") == f"{base}/norm_NR_ep2.5-350_eq0.75-200.h5")
    doi_registry = os.path.join(tmp, "doi.json")
    with open(doi_registry, "w") as f:
        json.dump({"tables": {}}, f)
    nt.register([table], doi="10.5281/zenodo.1234567", registry=doi_registry)
    check("register --doi writes the record's direct https file URL",
          nt.load_registry(doi_registry)["NR_ep2.5-350_eq0.75-200"]["url"]
          == "https://zenodo.org/records/1234567/files/norm_NR_ep2.5-350_eq0.75-200.h5?download=1")
    check("register --doi refuses a non-Zenodo DOI",
          raises(ValueError, lambda: nt.register([table], doi="10.1000/xyz", registry=doi_registry)))
    check("register refuses a file not named norm_<name>.h5",
          raises(ValueError, lambda: nt.register([registry], url_base=base, registry=registry)))

    path = nt.fetch("NR_ep2.5-350_eq0.75-200", registry=registry)
    check("fetch downloads into $BAND_TABLES_DIR", os.path.dirname(path) == cache and nt.sha256(path) == nt.sha256(table))
    check("fetch by file name works too", nt.fetch("norm_NR_ep2.5-350_eq0.75-200.h5", registry=registry) == path)

    server.shutdown()
    server.server_close()          # refuse connections from here on: "no network"
    check("cached table is used with no network", nt.fetch("NR_ep2.5-350_eq0.75-200", registry=registry) == path)

    with open(path, "ab") as f:
        f.write(b"corrupted")
    check("a cached file with the wrong hash is not returned (re-download fails offline)",
          raises(Exception, lambda: nt.fetch("NR_ep2.5-350_eq0.75-200", registry=registry)))

    try:
        nt.fetch("ER_nonexistent", registry=registry)
        msg = ""
    except KeyError as err:
        msg = str(err)
    check("unknown name: KeyError listing the registered tables", "NR_ep2.5-350_eq0.75-200" in msg)
    check("shipped registry parses", isinstance(nt.load_registry(), dict))
finally:
    shutil.rmtree(tmp)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
