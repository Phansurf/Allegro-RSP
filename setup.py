"""
Allegro-RSP setup — patches installed nequip/allegro with RSP modifications.

Usage:
    pip install -e . --no-build-isolation --no-deps

Prerequisites: nequip and allegro must already be installed (editable or not).
This script locates the installed package directories and copies RSP's
modified files over them, enabling the Reciprocal Space Potential extension.
"""
import shutil
from pathlib import Path
from setuptools import setup

RSP_DIR = Path(__file__).parent


def patch_installed_packages():
    """Copy RSP patch files into installed nequip/allegro."""
    try:
        import nequip
        import allegro
    except ImportError:
        print("Allegro-RSP: nequip/allegro not installed, skipping patch.")
        return

    targets = [
        ("nequip", RSP_DIR / "nequip", Path(nequip.__file__).parent),
        ("allegro", RSP_DIR / "allegro", Path(allegro.__file__).parent),
    ]

    patched = 0
    for pkg_name, src_root, dst_root in targets:
        if src_root.resolve() == dst_root.resolve():
            continue  # editable install pointing at itself, skip
        for src in src_root.rglob("*.py"):
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            patched += 1
            print(f"  Patched {pkg_name}/{rel}")

    print(f"Allegro-RSP: {patched} files patched.")


patch_installed_packages()
setup()
