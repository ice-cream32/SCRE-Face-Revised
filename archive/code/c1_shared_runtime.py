"""Export and run the frozen C1_SCORE with shared DenseNet tensors and lazy voting.

Export reads only original neural checkpoints and source calibration metadata.
Prediction is a separate, explicit operation; it never reads labels for fitting.
The exported directory contains its own network definitions and runtime copy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy.special import expit, logit
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


ROOT = Path('${ORIGINAL_WORKSPACE}')
DEFAULT_OUT = ROOT / 'outputs/C1_SCORE消融与简化/compact'
FORMAT = 'c1-shared-runtime-v1'
EXPECTED_PARAMETERS = 15_294_620
BRANCHES = ('c0', 'pair1', 'pair2', 'pair3')
KINDS = ('resnet8', 'hires19', 'densenet121')
SMALL_SEEDS = (42, 7, 2026)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes and bytes, in the supplied state-dict order."""
    digest = hashlib.sha256()
    for name, value in state.items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + b'\0')
        digest.update(str(tuple(value.shape)).encode() + b'\0' + value.numpy().tobytes())
    return digest.hexdigest()


def _networks(path: Path):
    spec = importlib.util.spec_from_file_location('_c1_compact_networks', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _checkpoint(branch: str, kind: str, seed: int) -> Path:
    if branch == 'c0':
        return ROOT / 'outputs/七网络原图通道与条件蒸馏/baseline_models' / f'full_{kind}_{seed}_base_stage2.pt'
    return ROOT / 'outputs/A1-A4模糊鲁棒性对照/models' / f'fold{branch[-1]}_A1_{kind}_{seed}.pt'


def _dense_parts(state: dict[str, torch.Tensor]):
    prefix = {
        key.removeprefix('features.'): value for key, value in state.items()
        if key.startswith('features.')
        and not key.startswith(('features.denseblock4.', 'features.norm5.'))
    }
    norm = {key.removeprefix('features.norm5.'): value for key, value in state.items()
            if key.startswith('features.norm5.')}
    tail = {key.removeprefix('features.denseblock4.'): value for key, value in state.items()
            if key.startswith('features.denseblock4.')}
    head = {key.removeprefix('classifier.'): value for key, value in state.items()
            if key.startswith('classifier.')}
    return prefix, norm, tail, head


class _CompactModel(nn.Module):
    def __init__(self, network_module):
        super().__init__()
        self.small = nn.ModuleDict()
        self.dense_tails = nn.ModuleDict()
        for branch in BRANCHES:
            for kind in KINDS[:2]:
                for seed in SMALL_SEEDS:
                    self.small[f'{branch}_{kind}_{seed}'] = (
                        network_module.Res8() if kind == 'resnet8' else network_module.HiRes19())
            dense = models.densenet121(weights=None)
            dense.classifier = nn.Linear(dense.classifier.in_features, 1)
            if branch == 'c0':
                # Keep names (conv0,...,transition3), not numeric sequential keys.
                from collections import OrderedDict
                self.prefix = nn.Sequential(OrderedDict(list(dense.features.named_children())[:-2]))
                self.norm5 = dense.features.norm5
            self.dense_tails[branch] = nn.ModuleDict({
                'block4': dense.features.denseblock4,
                'classifier': dense.classifier,
            })
        self.eval()
        self.requires_grad_(False)


def _compiled_gate(config: dict) -> dict:
    gate = config['gate']
    coefficient = np.asarray(gate['coefficients'], dtype=np.float64)
    mean = np.asarray(gate['mean'], dtype=np.float64)
    scale = np.asarray(gate['scale'], dtype=np.float64)
    assert gate['dimensions'] == 3 and np.all(scale > 0)
    threshold = float(config['threshold'])
    assert 0 < threshold < 1
    weight = coefficient / scale
    centered_bias = float(gate['intercept'] - mean @ weight - logit(threshold))
    return {
        'weights': weight.tolist(), 'decision_bias': centered_bias,
        'source_config': config,
        'fallback_rule': 'abs(margin)<=1e-10*(1+abs(L)@abs(weights)+abs(decision_bias))',
    }


def export_compact(outdir: str | Path = DEFAULT_OUT) -> dict:
    """Atomically create a portable CPU checkpoint, without any image/label reads."""
    outdir = Path(outdir).resolve()
    network_source = ROOT / 'outputs/七网络定向鲁棒优化/networks.py'
    config_source = ROOT / 'outputs/C组保留人脸与校准救援/部署校准器与阈值.json'
    config_all = json.loads(config_source.read_text(encoding='utf-8'))
    configs = [value['C1_SCORE'] for value in config_all]
    assert len(configs) == 3
    sources = []
    for branch in BRANCHES:
        for kind in KINDS:
            for seed in ((42,) if kind == 'densenet121' else SMALL_SEEDS):
                path = _checkpoint(branch, kind, seed)
                sources.append({'branch': branch, 'kind': kind, 'seed': seed,
                                'file': str(path), 'sha256': sha256(path)})
    signature = {
        'format': FORMAT, 'sources': sources,
        'network_source_sha256': sha256(network_source),
        'calibration_source_sha256': sha256(config_source),
        'runtime_source_sha256': sha256(Path(__file__)),
    }
    if outdir.exists():
        existing = json.loads((outdir / 'metadata.json').read_text(encoding='utf-8'))
        assert existing['export_signature'] == signature, 'Existing compact export has different inputs/code.'
        for name, digest in existing['portable_files_sha256'].items():
            assert sha256(outdir / name) == digest, f'Export file changed: {name}'
        return existing

    outdir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.compact-export-', dir=outdir.parent))
    try:
        shutil.copyfile(network_source, temp / 'networks.py')
        shutil.copyfile(Path(__file__), temp / 'c1_shared_runtime.py')
        module = _networks(temp / 'networks.py')
        # Constructor initialization remains CPU-only and preserves caller RNG.
        with torch.random.fork_rng(devices=[]):
            compact = _CompactModel(module)
        first_prefix = first_norm = None
        shared = []
        for source in sources:
            path = Path(source['file'])
            ck = torch.load(path, map_location='cpu', weights_only=True)
            state = ck['state_dict']
            if source['kind'] != 'densenet121':
                compact.small[f"{source['branch']}_{source['kind']}_{source['seed']}"].load_state_dict(state)
                continue
            prefix, norm, tail, head = _dense_parts(state)
            if first_prefix is None:
                first_prefix, first_norm = prefix, norm
                compact.prefix.load_state_dict(prefix)
                compact.norm5.load_state_dict(norm)
            else:
                assert prefix.keys() == first_prefix.keys() and norm.keys() == first_norm.keys()
                assert tensor_hash(prefix) == tensor_hash(first_prefix)
                assert tensor_hash(norm) == tensor_hash(first_norm)
            compact.dense_tails[source['branch']]['block4'].load_state_dict(tail)
            compact.dense_tails[source['branch']]['classifier'].load_state_dict(head)
            shared.append({'branch': source['branch'], 'prefix_tensor_sha256': tensor_hash(prefix),
                           'norm5_tensor_sha256': tensor_hash(norm)})
        parameter_count = sum(value.numel() for value in compact.parameters())
        assert parameter_count == EXPECTED_PARAMETERS, parameter_count
        state = {key: value.detach().cpu().contiguous() for key, value in compact.state_dict().items()}
        compiled = [_compiled_gate(config) for config in configs]
        checkpoint_path = temp / 'compact_model.pt'
        torch.save({'format': FORMAT, 'state_dict': state, 'compiled_gates': compiled}, checkpoint_path)
        metadata = {
            'format': FORMAT, 'export_signature': signature,
            'production_parameters_before_sharing': 29_681_948,
            'production_parameters_after_sharing': parameter_count,
            'frozen_shared_parameters': 4_795_776,
            'small_networks': 24, 'distinct_dense_tails': 4,
            'family_order': list(KINDS), 'seed_order': list(SMALL_SEEDS),
            'reference_threshold': 0.37, 'votes_required': 2,
            'prefix_output_per_image': [512, 4, 4], 'shared_tensor_audit': shared,
            'compiled_gates': compiled,
            'original_calibration_file': str(config_source),
            'float_rules': 'Network/sigmoid/seed means/family means/L extraction float32; fusion float64; critical-margin original-gate fallback.',
            'lazy_rule': 'C0 all images; pair1+pair2 on C0 negatives; pair3 only when their votes disagree.',
            'no_images_or_labels_read_at_export': True,
            'portable_files_sha256': {name: sha256(temp / name) for name in
                                      ['compact_model.pt', 'networks.py', 'c1_shared_runtime.py']},
            'libraries': {'torch': torch.__version__, 'numpy': np.__version__},
            'usage': {
                'python': 'runtime=SharedRuntime(model_dir,device); labels,stats=runtime.predict(images_cpu_N1x19x19)',
                'command': 'python c1_shared_runtime.py predict --model-dir . --input-mat INPUT.mat --mat-key test_data --reference-txt REFERENCE.txt --output-dir predictions',
            },
        }
        (temp / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, outdir)
        return metadata
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


class SharedRuntime:
    """One shared prefix/norm5, four DenseNet tails, and actual subset inference."""
    def __init__(self, model_dir: str | Path = DEFAULT_OUT, device: str | None = None):
        self.model_dir = Path(model_dir).resolve()
        self.metadata = json.loads((self.model_dir / 'metadata.json').read_text(encoding='utf-8'))
        assert self.metadata['format'] == FORMAT
        for name, digest in self.metadata['portable_files_sha256'].items():
            assert sha256(self.model_dir / name) == digest, f'Compact artifact changed: {name}'
        self.device = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
        if self.device.type not in ('cpu', 'mps'):
            raise ValueError('This runtime supports CPU and MPS.')
        self.networks = _networks(self.model_dir / 'networks.py')
        with torch.random.fork_rng(devices=[]):
            self.model = _CompactModel(self.networks)
        checkpoint = torch.load(self.model_dir / 'compact_model.pt', map_location='cpu', weights_only=True)
        assert checkpoint['format'] == FORMAT
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval().requires_grad_(False)
        self.compiled = checkpoint['compiled_gates']
        assert self.compiled == self.metadata['compiled_gates']
        self.parameter_count = sum(value.numel() for value in self.model.parameters())
        assert self.parameter_count == EXPECTED_PARAMETERS

    def _sync(self):
        if self.device.type == 'mps':
            torch.mps.synchronize()

    def _small(self, branch, kind, images, ids, batch_size, stats):
        output = []
        for seed in SMALL_SEEDS:
            net = self.model.small[f'{branch}_{kind}_{seed}']
            pieces = []
            for indices in torch.as_tensor(ids, dtype=torch.long).split(batch_size):
                view = images[indices].to(self.device)
                pieces.append(net(self.networks.prep(view, kind)).reshape(-1).sigmoid().cpu().numpy())
            output.append(np.concatenate(pieces).astype(np.float32, copy=False))
            stats['small_network_image_evaluations'] += len(ids)
            stats['small_calls'].append({'branch': branch, 'kind': kind, 'seed': seed, 'images': len(ids)})
        # Keep the original NumPy float32 seed mean, not logit averaging.
        return np.mean(output, axis=0, dtype=np.float32)

    def _dense(self, branch, encoded_cpu, ids, batch_size, stats):
        tail = self.model.dense_tails[branch]
        pieces = []
        for indices in torch.as_tensor(ids, dtype=torch.long).split(batch_size):
            value = tail['block4'](encoded_cpu[indices].to(self.device))
            value = F.relu(self.model.norm5(value), inplace=True)
            value = F.adaptive_avg_pool2d(value, (1, 1)).flatten(1)
            pieces.append(tail['classifier'](value).reshape(-1).sigmoid().cpu().numpy())
        stats['dense_tail_image_evaluations'] += len(ids)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def _families(self, branch, images, encoded_cpu, ids, batch_size, stats):
        values = [self._small(branch, kind, images, ids, batch_size, stats) for kind in KINDS[:2]]
        values.append(self._dense(branch, encoded_cpu, ids, batch_size, stats))
        stats['branch_images'][branch] = int(len(ids))
        return np.stack(values, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _votes(families, compiled):
        # Original c.features computes clip/log in float32, then promotes to
        # float64 when concatenating quality features, even for C1's 3D gate.
        clipped = np.asarray(families, dtype=np.float32).clip(1e-5, 1-1e-5)
        values = np.log(clipped / (1-clipped)).clip(-8, 8).astype(np.float64)
        weights = np.asarray(compiled['weights'], dtype=np.float64)
        bias = float(compiled['decision_bias'])
        margin = values @ weights + bias
        decision = margin >= 0
        tolerance = 1e-10 * (1 + np.abs(values) @ np.abs(weights) + abs(bias))
        close = np.abs(margin) <= tolerance
        if np.any(close):
            config = compiled['source_config']
            gate = config['gate']
            original = ((values[close] - gate['mean']) / gate['scale']) @ np.asarray(gate['coefficients'])
            original = expit(original + gate['intercept'])
            decision[close] = original >= config['threshold']
        return decision, int(close.sum()), float(np.min(np.abs(margin))) if len(margin) else None

    @torch.no_grad()
    def predict(self, images: torch.Tensor, batch_size: int = 64):
        """Return model labels (+1/-1) and measured work counts; no data overrides."""
        if not isinstance(images, torch.Tensor) or images.device.type != 'cpu':
            raise ValueError('Expected a CPU torch.Tensor, N x 1 x 19 x 19.')
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 19, 19):
            raise ValueError(f'Wrong input shape: {tuple(images.shape)}')
        if images.dtype != torch.float32:
            raise ValueError('Expected float32 inputs to preserve the original preprocessing.')
        if batch_size < 1 or not torch.isfinite(images).all() or (len(images) and (images.min() < 0 or images.max() > 1)):
            raise ValueError('Expected finite [0,1] pixels and positive batch_size.')
        self.model.eval()
        count = len(images)
        stats = {
            'format': FORMAT, 'device': str(self.device), 'images': count, 'batch_size': batch_size,
            'parameter_count': self.parameter_count, 'prefix_images': 0,
            'branch_images': {branch: 0 for branch in BRANCHES}, 'small_calls': [],
            'small_network_image_evaluations': 0, 'dense_tail_image_evaluations': 0,
            'critical_original_gate_fallbacks': {branch: 0 for branch in BRANCHES[1:]},
            'minimum_absolute_compiled_margin': {},
            'reference_positive_shortcuts': 0, 'two_vote_shortcuts': 0,
            'manual_overrides': False,
        }
        self._sync()
        started = time.monotonic()
        if not count:
            stats['seconds'] = 0.0
            return np.empty(0, dtype=np.int64), stats
        encoded = []
        for batch in images.split(batch_size):
            value = self.model.prefix(self.networks.prep(batch.to(self.device), 'densenet121'))
            assert tuple(value.shape[1:]) == (512, 4, 4)
            encoded.append(value.cpu())
        encoded_cpu = torch.cat(encoded)
        stats['prefix_images'] = count
        stats['prefix_encoded_bytes'] = encoded_cpu.numel() * encoded_cpu.element_size()
        all_ids = np.arange(count)
        reference_families = self._families('c0', images, encoded_cpu, all_ids, batch_size, stats)
        reference = np.mean(reference_families.T, axis=0, dtype=np.float32).astype(np.float64)
        positive = reference >= 0.37
        stats['reference_positive_shortcuts'] = int(positive.sum())
        pending = np.flatnonzero(~positive)
        if len(pending):
            first_two = []
            for branch, config in zip(BRANCHES[1:3], self.compiled[:2]):
                families = self._families(branch, images, encoded_cpu, pending, batch_size, stats)
                votes, fallback_count, minimum = self._votes(families, config)
                first_two.append(votes)
                stats['critical_original_gate_fallbacks'][branch] = fallback_count
                stats['minimum_absolute_compiled_margin'][branch] = minimum
            same = first_two[0] == first_two[1]
            stats['two_vote_shortcuts'] = int(same.sum())
            positive[pending[same]] = first_two[0][same]
            disputed = pending[~same]
            if len(disputed):
                families = self._families('pair3', images, encoded_cpu, disputed, batch_size, stats)
                vote, fallback_count, minimum = self._votes(families, self.compiled[2])
                positive[disputed] = vote
                stats['critical_original_gate_fallbacks']['pair3'] = fallback_count
                stats['minimum_absolute_compiled_margin']['pair3'] = minimum
        self._sync()
        stats['seconds'] = time.monotonic() - started
        stats['labels_face'] = int(positive.sum())
        stats['labels_nonface'] = int((~positive).sum())
        stats['dense_prefix_reuse_saved_image_evaluations'] = sum(stats['branch_images'].values()) - count
        return np.where(positive, 1, -1).astype(np.int64), stats


def _default_model_dir():
    local = Path(__file__).resolve().parent
    return local if (local / 'metadata.json').exists() else DEFAULT_OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='CPU-only export; no image or label reads.')
    export.add_argument('--model-dir', type=Path, default=DEFAULT_OUT)
    infer = commands.add_parser('predict', help='Explicit frozen prediction, separate from export.')
    infer.add_argument('--model-dir', type=Path, default=_default_model_dir())
    infer.add_argument('--input-mat', type=Path, required=True)
    infer.add_argument('--mat-key', default='test_data')
    infer.add_argument('--reference-txt', type=Path)
    infer.add_argument('--output-dir', type=Path, required=True)
    infer.add_argument('--expected-count', type=int, default=420)
    infer.add_argument('--device', choices=['cpu', 'mps'])
    infer.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.command == 'export':
        metadata = export_compact(args.model_dir)
        print(json.dumps({'exported': str(args.model_dir), 'parameters': metadata['production_parameters_after_sharing'],
                          'images_or_labels_read': False}, ensure_ascii=False))
        return
    from scipy.io import loadmat
    matrix = loadmat(args.input_mat)[args.mat_key]
    if matrix.shape != (args.expected_count, 361):
        raise ValueError(f'Expected {(args.expected_count,361)}, got {matrix.shape}.')
    images = torch.from_numpy(matrix.reshape(-1,19,19).transpose(0,2,1).copy()).float()[:,None]
    runtime = SharedRuntime(args.model_dir, args.device)
    labels, stats = runtime.predict(images, args.batch_size)
    stats['input_file_sha256'] = sha256(args.input_mat)
    if args.reference_txt:
        reference = np.loadtxt(args.reference_txt, dtype=np.int64)
        np.testing.assert_array_equal(reference[:,0], np.arange(1,len(labels)+1))
        assert reference.shape == (len(labels),2) and np.isin(reference[:,1],[-1,1]).all()
        changed = np.flatnonzero(labels != reference[:,1]) + 1
        stats['reference_file_sha256'] = sha256(args.reference_txt)
        stats['reference_different_ids'] = changed.tolist()
        stats['reference_identical'] = len(changed) == 0
        if len(changed):
            raise AssertionError('Compact model differs from reference: ' + json.dumps(stats, ensure_ascii=False))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output = args.output_dir / 'result_C1_COMPACT.txt'
    output.write_text(''.join(f'{index} {int(label)}\n' for index,label in enumerate(labels,1)),encoding='ascii')
    stats['prediction_sha256'] = sha256(output)
    (args.output_dir/'compact_runtime_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
