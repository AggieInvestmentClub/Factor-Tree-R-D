"""Portable v39 handoff. All mutable work and licensed inputs stay outside Git."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VERSION = Path('current/v39_forward_validation')
MARKER = '.v39-workspace.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def manifest_files(profile):
    profiles = {'replay'} if profile == 'replay' else {'replay', 'rebuild'}
    return [x for x in read(ROOT / 'data_contract/input_manifest.json')['files']
            if profiles.intersection(x['profiles'])]


def inside(path, root):
    return path == root or root in path.parents


def external_path(raw):
    path = Path(raw).expanduser().resolve()
    # Reject the package AND its repository, preventing accidental large-data commits.
    checkout = next((p for p in ROOT.parents if (p / '.git').exists()), ROOT)
    if inside(path, checkout) or inside(checkout, path):
        raise ValueError('Input/work directory must be separate from the Git checkout.')
    return path


def verify_package():
    rows = read(ROOT / 'provenance/source_manifest.json')['files']
    for row in rows:
        path = ROOT / 'engine' / row['path']
        if sha(path) != row['distributed_sha256']:
            raise ValueError(f'Source hash mismatch: {row["path"]}')
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=row['path'])
    if (ROOT / 'provenance/package_manifest.json').exists():
        inventory = read(ROOT / 'provenance/package_manifest.json')['files']
        expected = {row['path'] for row in inventory} | {'provenance/package_manifest.json'}
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        if actual != expected:
            raise ValueError('Package inventory changed; missing or unexpected files: ' + str(sorted(actual ^ expected)))
        for row in inventory:
            if sha(ROOT / row['path']) != row['sha256']:
                raise ValueError(f'Package hash mismatch: {row["path"]}')
    print(f'PACKAGE_OK: {len(rows)} engine source hashes and syntax checked', flush=True)


def check_data(input_root, profile):
    files = manifest_files(profile)
    errors = []
    for row in files:
        path = input_root / row['path']
        # Resolve each file as well; a manifest cannot redirect a read outside input_root.
        if not inside(path.resolve(), input_root):
            errors.append(row['path'] + ': path escapes input root')
        elif not path.is_file():
            errors.append(row['path'] + ': missing')
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            errors.append(row['path'] + ': size/hash mismatch')
    if errors:
        raise ValueError('INPUT CHECK FAILED\n' + '\n'.join(errors))
    result = {'passed': True, 'profile': profile, 'files': len(files),
              'bytes': sum(x['bytes'] for x in files), 'all_sha256_match': True,
              'manifest_sha256': sha(ROOT / 'data_contract/input_manifest.json')}
    print(json.dumps(result), flush=True)
    return result


def stage(args):
    verify_package()
    src, dest = external_path(args.input_root), external_path(args.work_root)
    if inside(src, dest) or inside(dest, src):
        raise ValueError('Work and input roots must be separate directories.')
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Use a new/empty work directory; existing work is never overwritten.')
    report = check_data(src, args.profile)
    dest.mkdir(parents=True, exist_ok=True)
    for family in ('current', 'prior'):
        shutil.copytree(ROOT / 'engine' / family, dest / family)
    omitted = []
    for row in manifest_files(args.profile):
        rel = row['path']
        if args.refit and any(rel.endswith(f'models/log_growth_{year}{ext}')
                             for year in (2025, 2026) for ext in ('.pt', '.json')):
            omitted.append(rel)
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
    write(dest / MARKER, {'schema_version': 1, 'profile': args.profile,
                         'input_check': report, 'omitted_for_refit': omitted,
                         'source_manifest_sha256': sha(ROOT / 'provenance/source_manifest.json')})
    print('STAGED: physical copies; original inputs unchanged', flush=True)


def workspace(raw):
    dest = external_path(raw)
    if not (dest / MARKER).is_file():
        raise ValueError('Not a staged v39 workspace; run stage first.')
    if read(dest / MARKER)['source_manifest_sha256'] != sha(ROOT / 'provenance/source_manifest.json'):
        raise ValueError('Workspace was staged with a different engine source manifest.')
    return dest


def limit_resources():
    # Applied only to this launcher and its children; does not alter other processes.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        import psutil
        proc = psutil.Process()
        if hasattr(proc, 'cpu_affinity'):
            allowed = proc.cpu_affinity()
            if allowed:
                proc.cpu_affinity([allowed[-1]])
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except (ImportError, AttributeError, OSError):
        pass


def run(args):
    dest = workspace(args.work_root)
    state = read(dest / MARKER)
    base = dest / VERSION
    steps = {
        'features': [('features.py',)],
        'labels': [('labels.py',)],
        'gate': [('gate.py',)],
        'arrays': [('build_arrays.py',)],
        'fit': [('portfolio.py', 'fit')],
        'targets': [('portfolio.py', 'prepare')],
        'accounts': [('portfolio.py', 'accounts')],
        'replay': [('portfolio.py', 'prepare'), ('portfolio.py', 'accounts')],
        'rebuild': [('features.py',), ('labels.py',), ('gate.py',),
                    ('build_arrays.py',), ('portfolio.py', 'fit'),
                    ('portfolio.py', 'prepare'), ('portfolio.py', 'accounts')],
    }
    if args.stage in {'features', 'labels', 'gate', 'arrays', 'rebuild'} and state['profile'] != 'rebuild':
        raise ValueError('This stage requires the rebuild input profile.')
    if args.stage in {'fit', 'rebuild'}:
        if any((base / f'models/log_growth_{year}.json').exists() for year in (2025, 2026)):
            raise ValueError('Cached models exist. Stage a new workspace with --refit for an actual refit.')
    if args.stage in {'replay', 'rebuild', 'accounts'}:
        if any((base / f'accounts/{arm}_complete.json').exists() for arm in ('benchmark', 'vol2sigma')):
            raise ValueError('Accounts already exist. Use compare-reference or stage a new workspace.')
    limit_resources()
    for command in steps[args.stage]:
        print('RUN:', ' '.join(command), flush=True)
        subprocess.run([sys.executable, '-u', str(base / 'src' / command[0]), *command[1:]],
                       cwd=dest, check=True)
    print('RUN_COMPLETE', flush=True)


def compare_reference(args):
    limit_resources()
    import numpy as np
    import pandas as pd
    import torch
    dest = workspace(args.work_root)
    base = dest / VERSION
    report = {'passed': True, 'comparison': 'All account ledger columns and numeric model states',
              'absolute_tolerance': 1e-10, 'accounts': {}, 'checkpoints': {},
              'conditional_valuations_remain': True, 'pristine_holdout': False}
    for arm in ('benchmark', 'vol2sigma'):
        expected = pd.read_csv(ROOT / f'reference/{arm}_ledger.csv.gz', index_col=0)
        actual = pd.read_csv(base / f'accounts/{arm}_ledger.csv.gz', index_col=0)
        pd.testing.assert_index_equal(expected.index, actual.index)
        pd.testing.assert_index_equal(expected.columns, actual.columns)
        numeric = expected.select_dtypes(include='number').columns
        errors = {}
        for col in numeric:
            a, b = expected[col].to_numpy(), actual[col].to_numpy()
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-10, equal_nan=True)
            diff = np.abs(a - b)
            errors[col] = float(np.max(diff[np.isfinite(diff)], initial=0))
        other = expected.columns.difference(numeric)
        pd.testing.assert_frame_equal(expected[other], actual[other])
        audit = read(base / f'accounts/{arm}_complete.json')['audit']
        report['accounts'][arm] = {'sessions': len(actual), 'start': actual.index[0],
                                   'end': actual.index[-1], 'max_absolute_error': max(errors.values()),
                                   'nav_max_absolute_error': errors['nav'],
                                   'return_max_absolute_error': errors['daily_return'],
                                   'arithmetic_passed': audit['arithmetic_passed'],
                                   'validated_performance': audit['validated_performance']}
    for year in (2024, 2025, 2026):
        expected = torch.load(ROOT / f'checkpoints/log_growth_{year}.pt', weights_only=True, map_location='cpu')
        actual = torch.load(base / f'models/log_growth_{year}.pt', weights_only=True, map_location='cpu')
        assert expected.keys() == actual.keys()
        errors = []
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=1e-10)
            errors.append(float((actual[key] - expected[key]).abs().max()))
        original = read(ROOT / f'checkpoints/log_growth_{year}.json')
        fitted = read(base / f'models/log_growth_{year}.json')
        assert fitted['selected_penalty'] == original['selected_penalty']
        report['checkpoints'][str(year)] = {'state_entries': len(expected),
                                          'max_absolute_error': max(errors),
                                          'selected_penalty': fitted['selected_penalty']}
    # Reports stay in the workspace. Publication is a separate deliberate copy.
    write(dest / 'reproduction.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify-package')
    check = sub.add_parser('check-data')
    check.add_argument('--input-root', required=True)
    check.add_argument('--profile', choices=['replay', 'rebuild'], default='replay')
    staging = sub.add_parser('stage')
    staging.add_argument('--input-root', required=True)
    staging.add_argument('--work-root', required=True)
    staging.add_argument('--profile', choices=['replay', 'rebuild'], default='replay')
    staging.add_argument('--refit', action='store_true', help='Omit 2025/26 checkpoints so fit actually trains them.')
    running = sub.add_parser('run')
    running.add_argument('--work-root', required=True)
    running.add_argument('--stage', choices=['features', 'labels', 'gate', 'arrays', 'fit',
                                           'targets', 'accounts', 'replay', 'rebuild'], required=True)
    compare = sub.add_parser('compare-reference')
    compare.add_argument('--work-root', required=True)
    args = parser.parse_args()
    if args.command == 'verify-package':
        verify_package()
    elif args.command == 'check-data':
        check_data(external_path(args.input_root), args.profile)
    elif args.command == 'stage':
        stage(args)
    elif args.command == 'run':
        run(args)
    elif args.command == 'compare-reference':
        compare_reference(args)


if __name__ == '__main__':
    main()
