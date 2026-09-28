"""Verify all files listed in docs/checksums.sha256 without third-party packages."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checksum_path = ROOT / 'docs' / 'checksums.sha256'
    if not checksum_path.is_file():
        print('Missing docs/checksums.sha256', file=sys.stderr)
        return 1
    checked = 0
    errors = []
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            errors.append(f'Missing or unsafe path: {relative}')
            continue
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            errors.append(f'Hash mismatch: {relative}')
        checked += 1
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'PASS: {checked} files match the handover SHA256 manifest.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
