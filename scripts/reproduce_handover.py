"""Reproduce the three batch stages in isolation and verify delivered results.

Only result/verification is written in the checkout. Canonical research results
are read as references, never overwritten. Run with the project's Conda Python.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import polars as pl
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    output = ROOT / 'result' / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {
        'status': 'running',
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('polars', 'numpy', 'matplotlib', 'pandas', 'streamlit', 'plotly', 'pyarrow')},
        'stages': [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix='pred-handover-reproduction-') as location:
            sandbox = Path(location)
            for directory in ('scripts', 'utils', 'static'):
                shutil.copytree(ROOT / directory, sandbox / directory,
                                ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT / 'config.json', sandbox / 'config.json')
            config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
            original = ROOT / config['source_path']
            target = sandbox / config['source_path']
            if not target.resolve().is_relative_to(sandbox.resolve()):
                raise ValueError('Reproduction requires a project-relative source_path')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            raw = pl.read_parquet(original)
            report['raw'] = {
                'sha256': digest(original), 'bytes': original.stat().st_size,
                'rows': raw.height, 'trading_days': raw['Date'].n_unique(),
                'contracts': raw['Contract'].n_unique(),
                'date_min': str(raw['Date'].min()), 'date_max': str(raw['Date'].max()),
                'schema': {k: str(v) for k, v in raw.schema.items()},
                'null_counts': raw.null_count().to_dicts()[0],
            }
            environment = dict(os.environ, MPLBACKEND='Agg')
            for script in ('analyze_segmented_score_drift.py',
                           'export_zero_centered_pred.py',
                           'analyze_adjusted_pred_returns.py'):
                before = time.time()
                completed = subprocess.run([sys.executable, str(sandbox/'scripts'/script)],
                                           cwd=sandbox, env=environment,
                                           text=True, capture_output=True)
                (output / (script.removesuffix('.py')+'.log')).write_text(
                    completed.stdout + completed.stderr, encoding='utf-8')
                report['stages'].append({'script': script, 'exit_code': completed.returncode,
                                         'seconds': round(time.time()-before, 3)})
                print(f'{script}: exit={completed.returncode}', flush=True)
                if completed.returncode:
                    raise RuntimeError(f'{script} failed; see result/verification logs')

            relative = Path('result/data/pred_eval_segmented_drift_adjusted.parquet')
            expected = pl.read_parquet(ROOT / relative)
            actual = pl.read_parquet(sandbox / relative)
            if actual.schema != expected.schema or actual.height != expected.height:
                raise AssertionError('Adjusted data schema or row count mismatch')
            other = [c for c in expected.columns if c != 'pred']
            if not actual.select(other).equals(expected.select(other)):
                raise AssertionError('Non-pred fields or row order mismatch')
            max_error = float((actual['pred'] - expected['pred']).abs().max())
            mismatches = int(((actual['pred'] < 0) != (expected['pred'] < 0)).sum())
            if max_error > 1e-12 or mismatches:
                raise AssertionError(f'Pred mismatch: max_error={max_error}, directions={mismatches}')
            report['adjusted'] = {'rows': actual.height, 'max_absolute_pred_difference': max_error,
                                  'direction_mismatches': mismatches, 'other_fields_exact': True,
                                  'delivered_sha256': digest(ROOT/relative),
                                  'regenerated_sha256': digest(sandbox/relative)}
            params_rel = Path('result/parameters/segmented_score_drift_selected_parameters.json')
            params = json.loads((sandbox/params_rel).read_text(encoding='utf-8'))
            reference = json.loads((ROOT/params_rel).read_text(encoding='utf-8'))
            if params != reference:
                raise AssertionError('Regenerated parameter JSON differs from delivered parameters')
            report['parameters_exact'] = True
            figures = sorted((sandbox/'result/visualizations').rglob('*.png'))
            reference_paths = {p.relative_to(ROOT/'result/visualizations').as_posix()
                               for p in (ROOT/'result/visualizations').rglob('*.png')}
            actual_paths = {p.relative_to(sandbox/'result/visualizations').as_posix() for p in figures}
            if reference_paths != actual_paths or len(figures) != 26:
                raise AssertionError('Generated figure set is incomplete')
            sizes = []
            for path in figures:
                with Image.open(path) as picture:
                    sizes.append({'path': path.relative_to(sandbox).as_posix(), 'size': picture.size})
                    picture.verify()
            report['figures'] = sizes
            report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['seconds'] = round(time.time()-started, 3)
        (output/'reproduction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('PASS: original data, selected parameters, all adjusted rows, directions and 26 PNGs verified.')


if __name__ == '__main__':
    main()
