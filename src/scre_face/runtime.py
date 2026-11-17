"""SCRE-Face S01 frozen inference. No training or reference-label input."""
from __future__ import annotations
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
DEFAULT_OUT=Path(__file__).resolve().parents[2]/'models/s01'
FORMAT='c1-shared-runtime-v1'
EXPECTED_PARAMETERS=15294620
BRANCHES=('c0','pair1','pair2','pair3')
KINDS=('resnet8','hires19','densenet121')
SMALL_SEEDS=(42,7,2026)
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
