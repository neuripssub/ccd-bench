"""
Mechanistic interpretability hooks: final-layer residual hidden states + optional SAELens features.
Captures tensors during the Act-phase forward pass and persists `.pt` payloads for offline analysis.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


def _decoder_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module] | None:
    """Resolve stacked decoder blocks for common HF causal LMs."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Qwen2, Mistral, etc.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    return None


def _last_layer_index(model: nn.Module) -> int:
    layers = _decoder_layers(model)
    if layers is None:
        raise ValueError("Unsupported architecture: could not find decoder layers")
    return len(layers) - 1


@dataclass
class ActivationStepRecord:
    """Metadata + paths written for one forward step."""

    step_id: str
    pt_path: str
    layer_index: int
    seq_len: int
    hidden_dim: int
    sae_dim: int | None = None


@dataclass
class MechanisticHookState:
    """Accumulates paths and summaries across a multi-step chain."""

    records: list[ActivationStepRecord] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("activation_dumps"))


class NoOpResidualHook:
    """Use when inference is remote (vLLM) and no local activations are captured."""

    def register(self) -> None:
        pass

    def remove(self) -> None:
        pass

    def clear_cache(self) -> None:
        pass

    @torch.no_grad()
    def flush_to_pt(self, **kwargs: Any) -> None:
        return None


class FinalLayerResidualHook:
    """
    Registers a forward hook on the final decoder layer.
    On each capture, saves residual stream (post-block hidden states) to `.pt` and optionally SAE codes.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_index: int | None = None,
        sae: nn.Module | None = None,
        output_dir: str | Path = "activation_dumps",
        token_slice: str = "last",
    ) -> None:
        self.model = model
        self.layer_index = layer_index if layer_index is not None else _last_layer_index(model)
        self.sae = sae
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token_slice = token_slice
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._last_hidden: torch.Tensor | None = None

    def _hook_fn(self, module: nn.Module, args: tuple, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            return
        self._last_hidden = hidden.detach()

    def register(self) -> None:
        layers = _decoder_layers(self.model)
        if layers is None:
            raise ValueError("Cannot attach hook: unknown model layout")
        layer = layers[self.layer_index]
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear_cache(self) -> None:
        self._last_hidden = None

    def _select_positions(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq, d)
        if self.token_slice == "all":
            return h
        if self.token_slice == "last":
            return h[:, -1:, :]
        if self.token_slice == "mean":
            return h.mean(dim=1, keepdim=True)
        raise ValueError(f"Unknown token_slice: {self.token_slice}")

    @torch.no_grad()
    def flush_to_pt(
        self,
        *,
        phase: str = "act",
        extra_meta: dict[str, Any] | None = None,
    ) -> ActivationStepRecord | None:
        if self._last_hidden is None:
            return None
        h = self._select_positions(self._last_hidden)
        step_id = str(uuid.uuid4())
        path = self.output_dir / f"{phase}_{step_id}.pt"

        sae_features: torch.Tensor | None = None
        sae_dim: int | None = None
        if self.sae is not None:
            enc = getattr(self.sae, "encode", None)
            if callable(enc):
                sae_features = enc(h)
            else:
                sae_features = self.sae(h)  # type: ignore[operator]
            if isinstance(sae_features, torch.Tensor):
                sae_dim = sae_features.shape[-1]

        payload: dict[str, Any] = {
            "step_id": step_id,
            "phase": phase,
            "layer_index": self.layer_index,
            "residual_stream_final_layer": h.cpu(),
            "sae_feature_activations": sae_features.cpu() if sae_features is not None else None,
            "meta": extra_meta or {},
        }
        torch.save(payload, path)
        rec = ActivationStepRecord(
            step_id=step_id,
            pt_path=str(path.resolve()),
            layer_index=self.layer_index,
            seq_len=int(h.shape[1]),
            hidden_dim=int(h.shape[2]),
            sae_dim=sae_dim,
        )
        self._last_hidden = None
        return rec


def try_load_sae(
    *,
    release: str | None,
    sae_id: str | None,
    device: str = "cpu",
    dtype: str | None = None,
) -> nn.Module | None:
    """Load SAELens SAE when release/sae_id are provided; return None on failure."""
    if not release or not sae_id:
        return None
    try:
        from sae_lens import SAE
    except ImportError:
        return None
    kwargs: dict[str, Any] = {"release": release, "sae_id": sae_id, "device": device}
    if dtype:
        kwargs["dtype"] = dtype
    try:
        out = SAE.from_pretrained(**kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out
    except Exception:
        return None


def build_hook_bundle(
    model: nn.Module,
    *,
    sae_release: str | None = None,
    sae_id: str | None = None,
    dump_dir: str | Path | None = None,
    layer_index: int | None = None,
    device: str | None = None,
) -> tuple[FinalLayerResidualHook, MechanisticHookState]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = try_load_sae(release=sae_release, sae_id=sae_id, device=dev)
    out = Path(dump_dir or os.environ.get("ACTIVATION_DUMP_DIR", "activation_dumps"))
    hook = FinalLayerResidualHook(
        model,
        layer_index=layer_index,
        sae=sae,
        output_dir=out,
    )
    state = MechanisticHookState(output_dir=out)
    return hook, state


def append_activation_cache(
    state: MechanisticHookState,
    record: ActivationStepRecord | None,
) -> MechanisticHookState:
    if record is not None:
        state.records.append(record)
    return state


def capture_residual_tlens(
    model_name: str,
    prompt: str,
    *,
    layer: int | None = None,
    device: str | None = None,
    dump_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Optional path: run a HookedTransformer forward with cache and persist final-layer residual.
    Requires `transformer_lens` and a compatible `model_name` registry entry.
    """
    from transformer_lens import HookedTransformer  # type: ignore[import-not-found]

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=dev)
    layer = layer if layer is not None else model.cfg.n_layers - 1
    _, cache = model.run_with_cache(prompt)
    hook_name = f"blocks.{layer}.hook_resid_post"
    resid = cache[hook_name].detach().cpu()
    out: dict[str, Any] = {
        "model_name": model_name,
        "hook": hook_name,
        "residual_stream_final_layer": resid[:, -1:, :],
    }
    if dump_path is not None:
        p = Path(dump_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, p)
        out["pt_path"] = str(p.resolve())
    return out


def spike_stats_from_pt(pt_path: str | Path) -> dict[str, float]:
    """Lightweight stats for correlation (no full SAE load)."""
    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(pt_path, map_location="cpu")
    feat = obj.get("sae_feature_activations")
    resid = obj.get("residual_stream_final_layer")
    out: dict[str, float] = {}
    if isinstance(resid, torch.Tensor):
        r = resid.float()
        out["residual_l2"] = float(r.norm().item())
        out["residual_max_abs"] = float(r.abs().max().item())
    if isinstance(feat, torch.Tensor):
        f = feat.float()
        out["sae_l0"] = float((f > 0).float().sum(dim=-1).mean().item())
        out["sae_max"] = float(f.max().item())
        out["sae_l1"] = float(f.abs().sum(dim=-1).mean().item())
    return out
