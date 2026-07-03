#!/usr/bin/env python3
"""
SETrack experiment config generator & training launcher.

Generate a new YAML config from an existing base config by overriding
EPOCH and LR_DROP_EPOCH, with optional --train and --profile hooks.

Usage:
    # Generate 50ep config only
    python tracking/run_setrack_exp.py \
      --base vitb_256_mae_setrack_base_fixmae_got10k_30ep \
      --epochs 50 --lr-drop 40

    # Generate 50ep config and profile
    python tracking/run_setrack_exp.py \
      --base vitb_256_mae_setrack_base_fixmae_got10k_30ep \
      --epochs 50 --lr-drop 40 --profile

    # Generate 50ep config and train
    python tracking/run_setrack_exp.py \
      --base vitb_256_mae_setrack_base_fixmae_got10k_30ep \
      --epochs 50 --lr-drop 40 --train
"""

import argparse
import os
import re
import sys
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prj_root():
    """Absolute path to the project root (one level above tracking/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _experiments_dir(script_name):
    return os.path.join(_prj_root(), 'experiments', script_name)


def _cfg_path(script_name, config_name):
    return os.path.join(_experiments_dir(script_name), f'{config_name}.yaml')


def _load_yaml(path):
    """Load a YAML file using the same PyYAML interface as the rest of the
    project. Returns a plain dict."""
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _dump_yaml(cfg, path):
    """Write a dict to a YAML file, preserving key order and readable
    formatting."""
    import yaml
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _yaml_content_equals(path_a, path_b):
    """Return True if two YAML files have identical parsed content."""
    try:
        return _load_yaml(path_a) == _load_yaml(path_b)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# auto-naming
# ---------------------------------------------------------------------------

_EP_SUFFIX_RE = re.compile(r'_\d+ep$', re.IGNORECASE)


def _auto_name(base_name, epochs):
    """Derive new config name from *base_name*.

    Rules (applied in order):
    1. If *base_name* ends with ``_{N}ep`` → replace with ``_{epochs}ep``.
    2. Otherwise → append ``_e{epochs}``.
    """
    if _EP_SUFFIX_RE.search(base_name):
        return _EP_SUFFIX_RE.sub(f'_{epochs}ep', base_name)
    return f'{base_name}_e{epochs}'


# ---------------------------------------------------------------------------
# safety checks
# ---------------------------------------------------------------------------

def _safety_check(cfg, base_name, args):
    """Run safety / informational checks on the loaded config.

    Returns a list of warning/error strings.  If any error is returned the
    script should abort.
    """
    errors = []
    warnings = []

    backbone = cfg.get('MODEL', {}).get('BACKBONE', {})

    use_cross = backbone.get('USE_CROSS_SEMANTIC', None)
    use_rim = backbone.get('USE_REDUNDANT_PRUNING', None)

    # ------------------------------------------------------------------
    # Always print their values for visibility
    # ------------------------------------------------------------------
    print(f'[SAFETY] USE_CROSS_SEMANTIC    = {use_cross}')
    print(f'[SAFETY] USE_REDUNDANT_PRUNING = {use_rim}')

    # ------------------------------------------------------------------
    # If base config name contains "base", enforce both are OFF
    # ------------------------------------------------------------------
    if 'base' in base_name.lower():
        if use_cross is not False:
            errors.append(
                f'Base config "{base_name}" has USE_CROSS_SEMANTIC={use_cross}, '
                f'but "base" configs MUST have USE_CROSS_SEMANTIC=False. '
                f'Please fix the base YAML or use a different base config.'
            )
        if use_rim is not False:
            errors.append(
                f'Base config "{base_name}" has USE_REDUNDANT_PRUNING={use_rim}, '
                f'but "base" configs MUST have USE_REDUNDANT_PRUNING=False. '
                f'Please fix the base YAML or use a different base config.'
            )

    # ------------------------------------------------------------------
    # Warn about non-standard pruning fill value
    # ------------------------------------------------------------------
    fill_val = backbone.get('PRUNING_FILL_VALUE', None)
    if fill_val == 'zero' and use_rim is True:
        warnings.append(
            'PRUNING_FILL_VALUE is "zero" while USE_REDUNDANT_PRUNING=True. '
            'This is typically an eval-only config, not a main training config.'
        )

    return errors, warnings


# ---------------------------------------------------------------------------
# config summary
# ---------------------------------------------------------------------------

def _print_summary(cfg, new_config_name, script_name, save_dir):
    print('\n' + '=' * 62)
    print('  NEW CONFIG SUMMARY')
    print('=' * 62)
    print(f'  config name       : {new_config_name}')
    print(f'  EPOCH             : {cfg["TRAIN"]["EPOCH"]}')
    print(f'  LR_DROP_EPOCH     : {cfg["TRAIN"]["LR_DROP_EPOCH"]}')
    print(f'  USE_CROSS_SEMANTIC: {cfg.get("MODEL", {}).get("BACKBONE", {}).get("USE_CROSS_SEMANTIC", "N/A")}')
    print(f'  USE_REDUNDANT_PRUNING: {cfg.get("MODEL", {}).get("BACKBONE", {}).get("USE_REDUNDANT_PRUNING", "N/A")}')
    print(f'  SAMPLE_PER_EPOCH  : {cfg.get("DATA", {}).get("TRAIN", {}).get("SAMPLE_PER_EPOCH", "N/A")}')
    print(f'  BATCH_SIZE        : {cfg["TRAIN"]["BATCH_SIZE"]}')
    print(f'  LR                : {cfg["TRAIN"]["LR"]}')
    print(f'  PRETRAIN_FILE     : {cfg.get("MODEL", {}).get("PRETRAIN_FILE", "N/A")}')
    print(f'  checkpoint dir    : {save_dir}/checkpoints/train/{script_name}/{new_config_name}')
    print('=' * 62 + '\n')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SETrack experiment config generator & training launcher.'
    )
    parser.add_argument('--script', type=str, default='setrack',
                        help='Script name (default: setrack)')
    parser.add_argument('--base', type=str, required=True,
                        help='Base config name WITHOUT .yaml, e.g. '
                             'vitb_256_mae_setrack_base_fixmae_got10k_30ep')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Target training epochs')
    parser.add_argument('--lr-drop', type=int, required=True,
                        help='LR drop epoch')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom new config name. Auto-generated if not given.')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Launch training after config generation')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Run profile_model.py after config generation')
    parser.add_argument('--save-dir', type=str, default='./output',
                        help='Checkpoint/log save directory (default: ./output)')
    parser.add_argument('--use-wandb', type=int, default=0, choices=[0, 1],
                        help='Use wandb (default: 0)')
    parser.add_argument('--use-lmdb', type=int, default=0, choices=[0, 1],
                        help='Use lmdb datasets (default: 0)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing target YAML if content differs')

    args = parser.parse_args()

    prj_root = _prj_root()
    os.chdir(prj_root)  # ensure cwd is project root for all subprocess calls

    # ------------------------------------------------------------------
    # 1. Locate base config
    # ------------------------------------------------------------------
    base_path = _cfg_path(args.script, args.base)
    if not os.path.isfile(base_path):
        exp_dir = _experiments_dir(args.script)
        print(f'[ERROR] Base config not found: {base_path}')
        if os.path.isdir(exp_dir):
            siblings = sorted(os.listdir(exp_dir))
            print(f'Available configs under experiments/{args.script}/:')
            for s in siblings:
                print(f'  {s}')
        else:
            print(f'experiments/{args.script}/ does not exist.')
        sys.exit(1)

    print(f'[INFO] Base config : {base_path}')

    # ------------------------------------------------------------------
    # 2. Read YAML
    # ------------------------------------------------------------------
    cfg = _load_yaml(base_path)

    # ------------------------------------------------------------------
    # 3. Safety checks
    # ------------------------------------------------------------------
    errors, warnings = _safety_check(cfg, args.base, args)
    if warnings:
        for w in warnings:
            print(f'[WARNING] {w}')
    if errors:
        for e in errors:
            print(f'[ERROR] {e}')
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Determine new config name
    # ------------------------------------------------------------------
    if args.name:
        new_name = args.name
    else:
        new_name = _auto_name(args.base, args.epochs)
    print(f'[INFO] New config  : {new_name}')

    # ------------------------------------------------------------------
    # 5. Modify fields
    # ------------------------------------------------------------------
    cfg['TRAIN']['EPOCH'] = args.epochs
    cfg['TRAIN']['LR_DROP_EPOCH'] = args.lr_drop

    # Also update TEST.EPOCH to match (used for checkpoint selection)
    if 'TEST' in cfg:
        cfg['TEST']['EPOCH'] = args.epochs

    # ------------------------------------------------------------------
    # 6. Write new YAML (with overwrite protection)
    # ------------------------------------------------------------------
    new_path = _cfg_path(args.script, new_name)
    new_dir = os.path.dirname(new_path)
    os.makedirs(new_dir, exist_ok=True)

    if os.path.isfile(new_path):
        # write temporary file for content comparison
        tmp_path = new_path + '.tmp'
        _dump_yaml(cfg, tmp_path)

        if _yaml_content_equals(new_path, tmp_path):
            print(f'[INFO] Target config already exists with identical content: {new_path}')
            os.remove(tmp_path)
        else:
            os.remove(tmp_path)
            if args.overwrite:
                print(f'[INFO] Overwriting existing config (--overwrite set): {new_path}')
            else:
                print(f'[ERROR] Target config already exists with DIFFERENT content: {new_path}')
                print(f'        Use --overwrite to force overwrite, or use --name to specify a different name.')
                sys.exit(1)

    if not os.path.isfile(new_path) or args.overwrite:
        _dump_yaml(cfg, new_path)
        print(f'[INFO] Written: {new_path}')

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    _print_summary(cfg, new_name, args.script, args.save_dir)

    # ------------------------------------------------------------------
    # 8. Profile (optional)
    # ------------------------------------------------------------------
    if args.profile:
        profile_script = os.path.join(prj_root, 'tracking', 'profile_model.py')
        cmd = [
            sys.executable, profile_script,
            '--script', args.script,
            '--config', new_name,
        ]
        print(f'[PROFILE] Running: {" ".join(cmd)}')
        result = subprocess.run(cmd, cwd=prj_root)
        if result.returncode != 0:
            print('[ERROR] profile_model.py failed. Check the output above.')
            sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # 9. Train (optional)
    # ------------------------------------------------------------------
    if args.train:
        train_script = os.path.join(prj_root, 'lib', 'train', 'run_training.py')
        cmd = [
            sys.executable, train_script,
            '--script', args.script,
            '--config', new_name,
            '--save_dir', args.save_dir,
            '--use_wandb', str(args.use_wandb),
            '--use_lmdb', str(args.use_lmdb),
        ]
        print(f'[TRAIN] Running: {" ".join(cmd)}')
        result = subprocess.run(cmd, cwd=prj_root)
        if result.returncode != 0:
            print('[ERROR] Training failed. Check the output above.')
            sys.exit(result.returncode)
    else:
        train_script = os.path.join(prj_root, 'lib', 'train', 'run_training.py')
        cmd = (
            f'python lib/train/run_training.py '
            f'--script {args.script} '
            f'--config {new_name} '
            f'--save_dir {args.save_dir} '
            f'--use_wandb {args.use_wandb} '
            f'--use_lmdb {args.use_lmdb}'
        )
        print(f'[DRY-RUN] To train, run:\n  {cmd}')


if __name__ == '__main__':
    main()
