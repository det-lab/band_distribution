"""
Published normalization tables: fetch by name, verified and cached.

The tables (python/normgrid.py, built by slurm/build_table.sh) are data, so
they live on Zenodo rather than in git; python/table_registry.json maps each
name to its URL and SHA-256.  fetch() downloads a table the first time with
pooch (https://www.fatiando.org/pooch/), checks the hash, and caches it;
later calls use the cached file with no network.  PpqPDF(ppqn_table=NAME)
calls it for any table argument that is not an existing file.

    import normtables
    path = normtables.fetch("NR_ep2.5-350_eq0.75-200")

Cache: $BAND_TABLES_DIR if set (the container sets it to /app/tables, where
its build fetched every registered table -- HPC compute nodes often have no
internet), else the user cache directory (~/.cache/band_distribution).

CLI (from the repository root):

  python python/normtables.py list
  python python/normtables.py fetch [NAME ...] [--all]
  python python/normtables.py register --doi 10.5281/zenodo.NNNN tables/norm_*.h5

`register` records files already uploaded to that Zenodo record (upload
first, then register and commit the registry); names come from the file
names, norm_<name>.h5.
"""

import argparse
import hashlib
import json
import os

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_registry.json")


def load_registry(path=REGISTRY):
    with open(path) as f:
        return json.load(f)["tables"]


def _cache_dir():
    import pooch
    return os.environ.get("BAND_TABLES_DIR") or pooch.os_cache("band_distribution")


def fetch(name, *, registry=REGISTRY):
    """Local path of the registered table `name` (downloading and verifying
    it the first time).  `name` is a registry name or its file name."""
    entries = load_registry(registry)
    by_file = {e["file"]: n for n, e in entries.items()}
    key = name if name in entries else by_file.get(os.path.basename(str(name)))
    if key is None:
        known = "\n  ".join(sorted(entries)) or "(none registered yet)"
        raise KeyError(f"{name!r} is neither an existing file nor a registered table; registered:\n  {known}")
    try:
        import pooch
    except ImportError:
        raise ImportError("fetching a registered table needs pooch (conda install -c conda-forge pooch); "
                          "or pass the table's file path") from None
    e = entries[key]
    fetcher = pooch.create(path=_cache_dir(), base_url="", env="BAND_TABLES_DIR",
                           registry={e["file"]: "sha256:" + e["sha256"]}, urls={e["file"]: e["url"]})
    return fetcher.fetch(e["file"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def zenodo_file_url(doi, fname):
    """The direct download URL of file `fname` in the Zenodo record with
    this version DOI (10.5281/zenodo.<record id>).  Plain https rather than
    pooch's doi: protocol, which older pooch versions lack; just as fixed,
    since a published record's files cannot change (and the hash is checked)."""
    prefix = "10.5281/zenodo."
    if not (doi.startswith(prefix) and doi[len(prefix):].isdigit()):
        raise ValueError(f"{doi!r} is not a Zenodo DOI (10.5281/zenodo.<record id>)")
    return f"https://zenodo.org/records/{doi[len(prefix):]}/files/{fname}?download=1"


def register(paths, *, doi=None, url_base=None, registry=REGISTRY):
    """Add (or update) registry entries for local table files that have been
    uploaded to the Zenodo record with version DOI `doi` (or to
    `url_base`/<file>)."""
    if (doi is None) == (url_base is None):
        raise ValueError("give exactly one of doi or url_base")
    with open(registry) as f:
        reg = json.load(f)
    for p in paths:
        fname = os.path.basename(p)
        if not (fname.startswith("norm_") and fname.endswith(".h5")):
            raise ValueError(f"{fname}: table files are named norm_<name>.h5")
        url = zenodo_file_url(doi, fname) if doi else f"{url_base.rstrip('/')}/{fname}"
        reg["tables"][fname[len("norm_"):-len(".h5")]] = {"file": fname, "url": url, "sha256": sha256(p)}
    reg["tables"] = dict(sorted(reg["tables"].items()))
    with open(registry, "w") as f:
        json.dump(reg, f, indent=2)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=REGISTRY, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="registered tables")
    p = sub.add_parser("fetch", help="download, verify and cache tables")
    p.add_argument("names", nargs="*"); p.add_argument("--all", action="store_true")
    p = sub.add_parser("register", help="add uploaded table files to the registry")
    p.add_argument("files", nargs="+")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--doi", help="the Zenodo record's version DOI (not the concept DOI), e.g. 10.5281/zenodo.1234567")
    g.add_argument("--url-base", help="any other location serving <url-base>/<file>")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for n, e in load_registry(a.registry).items():
            print(f"{n}\t{e['url']}")
    elif a.cmd == "fetch":
        names = list(load_registry(a.registry)) if a.all else a.names
        if not names and not a.all:
            ap.error("fetch: give table names, or --all")
        for n in names:
            print(fetch(n, registry=a.registry))
    elif a.cmd == "register":
        register(a.files, doi=a.doi, url_base=a.url_base, registry=a.registry)
        print(f"registered {len(a.files)} table(s) in {a.registry}")


if __name__ == "__main__":
    main()
