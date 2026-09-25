"""
Integrity check for raw tomograms before ``dloct.prepare_data``.

For every ``*.npy`` under the given directory:
  * the header parses and the file size equals header + shape × itemsize (catches truncated
    downloads),
  * the layout is ``(Z, X, Y)`` complex or ``(Z, X, Y, 2)`` real,
  * a few B-scans (first, middle, last) are finite and not all zero,
and every A/B channel pair has matching shape and dtype. Read-only.

    python scripts/check_data.py ~/dloct/raw/phase [--expect 29]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dloct.prepare_data import group_of  # noqa: E402


def check(path: Path):
    problems = []
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read = {(1, 0): np.lib.format.read_array_header_1_0}.get(version, np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read(f)
        header_len = f.tell()
    expected = header_len + int(np.prod(shape)) * dtype.itemsize
    actual = path.stat().st_size
    if actual != expected:
        problems.append(f"size {actual} != expected {expected} ({100 * actual / expected:.1f}% present): truncated?")
        return shape, dtype, problems
    if fortran:
        problems.append("Fortran order (unexpected)")
    is_complex = np.issubdtype(dtype, np.complexfloating)
    if not ((len(shape) == 3 and is_complex) or (len(shape) == 4 and shape[-1] == 2 and not is_complex)):
        problems.append(f"unexpected layout {shape} {dtype}")
        return shape, dtype, problems
    a = np.load(path, mmap_mode="r")
    for y in sorted({0, shape[2] // 2, shape[2] - 1}):
        b = np.asarray(a[:, :, y])
        if not np.isfinite(b).all():
            problems.append(f"non-finite values in B-scan y={y}")
        if not np.any(b):
            problems.append(f"B-scan y={y} is all zeros")
    return shape, dtype, problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--expect", type=int, default=None, help="expected number of files")
    args = p.parse_args()

    files = sorted(Path(args.dir).expanduser().glob("*.npy"))
    ok = True
    shapes = {}
    print(f"{'file':28s} {'shape':22s} {'dtype':10s} {'GB':>6s}  status")
    for f in files:
        try:
            shape, dtype, problems = check(f)
        except Exception as e:  # unreadable header, etc.
            shape, dtype, problems = "?", "?", [f"unreadable: {e}"]
        shapes[f.stem] = (tuple(shape) if shape != "?" else shape, str(dtype))
        shape = shape if shape != "?" else ("?",)
        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        ok &= not problems
        print(f"{f.name:28s} {str(tuple(shape)):22s} {str(dtype):10s} {f.stat().st_size / 1e9:6.2f}  {status}")

    groups = {}
    for stem in shapes:
        groups.setdefault(group_of("phase", stem), []).append(stem)
    print(f"\n{len(files)} files, {len(groups)} samples (A/B channels grouped):")
    for g, members in sorted(groups.items()):
        same = len({shapes[m] for m in members}) == 1
        ok &= same
        print(f"  {g.split('/', 1)[1]:22s} {', '.join(sorted(members))}{'' if same else '   <-- A/B shapes differ'}")

    if args.expect is not None and len(files) != args.expect:
        ok = False
        print(f"\nexpected {args.expect} files, found {len(files)}")
    print("\nALL OK" if ok else "\nPROBLEMS FOUND (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
