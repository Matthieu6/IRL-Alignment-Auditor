"""
RLHF Model Evaluation

Evaluates trained RLHF models on toxicity reduction using real-toxicity-prompts.
Supports both HuggingFace repos and local checkpoints.
"""

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from trl import AutoModelForCausalLMWithValueHead
from tqdm import tqdm


# ============================================================================
# Utilities
# ============================================================================

def is_local_path(p: str) -> bool:
    """Check if path is local directory."""
    try:
        return os.path.isdir(p)
    except Exception:
        return False


def pick_dtype():
    """Select best dtype for GPU/CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_for_filename(s: str) -> str:
    """Turn arbitrary string into safe filename token."""
    s = s.strip().strip("/\\")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] if len(s) > 120 else s


def make_root_tag(root: str) -> str:
    """Create compact identifier for model root."""
    if is_local_path(root):
        r = root.rstrip("/\\")
        parent = os.path.basename(os.path.dirname(r))
        base = os.path.basename(r)
        candidate = f"{parent}__{base}" if parent else base
    else:
        candidate = root
    return sanitize_for_filename(candidate)


# ============================================================================
# Model Loading
# ============================================================================

def load_tokenizer(model_id: str):
    """Load tokenizer with PPO-compatible settings."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def try_load_lm(model_id: str, torch_dtype):
    """Load RLHF model (with value head) or plain CausalLM."""
    kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    try:
        return AutoModelForCausalLMWithValueHead.from_pretrained(model_id, **kwargs)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def resolve_ctx_len(model_or_id, tokenizer) -> int:
    """Get model's context length."""
    try:
        cid = model_or_id if isinstance(model_or_id, str) else None
        cfg = AutoConfig.from_pretrained(cid) if cid else getattr(model_or_id, "config", None)
        mlen = int(getattr(cfg, "max_position_embeddings", tokenizer.model_max_length))
    except Exception:
        mlen = int(tokenizer.model_max_length)
    if mlen is None or mlen > 100_000_000:
        mlen = 4096
    return mlen


# ============================================================================
# Generation Configuration
# ============================================================================

def _extract_gen_from_flat(flat: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extract generation config from nested dict."""
    if not isinstance(flat, dict):
        return None
    candidates = []
    if "model" in flat and isinstance(flat["model"], dict):
        if "generation" in flat["model"]:
            candidates.append(flat["model"]["generation"])
    for key in ["generation", "generation_kwargs", "gen", "gen_kwargs"]:
        if key in flat and isinstance(flat[key], dict):
            candidates.append(flat[key])
    
    for g in candidates:
        if all(k in g for k in ["max_new_tokens", "do_sample", "top_p"]):
            return {
                "max_new_tokens": int(g["max_new_tokens"]),
                "do_sample": bool(g["do_sample"]),
                "top_p": float(g.get("top_p", 1.0)),
                "top_k": int(g.get("top_k", 0)),
                "temperature": float(g.get("temperature", 1.0)),
            }
    return None


def load_gen_cfg_from_any(repo_or_path_list: List[str], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Try to load generation config from rlhf_config.yaml."""
    for rid in repo_or_path_list:
        if is_local_path(rid):
            candidates = [
                os.path.join(rid, "rlhf_config.yaml"),
                os.path.join(rid, "config.yaml"),
                os.path.join(os.path.dirname(rid), "rlhf_config.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(rid)), "rlhf_config.yaml"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    try:
                        cfg = OmegaConf.load(c)
                        flat = OmegaConf.to_container(cfg, resolve=True)
                        out = _extract_gen_from_flat(flat)
                        if out is not None:
                            return out
                    except Exception:
                        pass
        else:
            for fname in ["rlhf_config.yaml", "config.yaml"]:
                try:
                    cfg_path = hf_hub_download(rid, filename=fname)
                    cfg = OmegaConf.load(cfg_path)
                    flat = OmegaConf.to_container(cfg, resolve=True)
                    out = _extract_gen_from_flat(flat)
                    if out is not None:
                        return out
                except Exception:
                    pass
    
    print("   ⚠️  Could not read rlhf_config.yaml; using config defaults.")
    return fallback


# ============================================================================
# Checkpoint Discovery
# ============================================================================

_EPOCH_RE = re.compile(r"checkpoint-epoch-(\d+)")

def _parse_epoch_from_path(p: str) -> int | None:
    """Extract epoch number from checkpoint path."""
    m = _EPOCH_RE.search(os.path.basename(p.rstrip("/")))
    return int(m.group(1)) if m else None


def discover_checkpoints(trained_root: str, epochs: List[int]) -> List[Tuple[str, str]]:
    """Discover checkpoint paths/IDs to evaluate."""
    out: List[Tuple[str, str]] = []
    
    if not is_local_path(trained_root):
        for e in epochs:
            out.append((f"epoch_{e}", f"{trained_root}-checkpoint-epoch-{e}"))
        return out
    
    root = os.path.abspath(trained_root)
    ckpts_dir = os.path.join(root, "checkpoints")
    
    if os.path.isdir(ckpts_dir):
        for e in epochs:
            p = os.path.join(ckpts_dir, f"checkpoint-epoch-{e}")
            if os.path.isdir(p):
                out.append((f"epoch_{e}", p))
        return out
    
    if os.path.isdir(root):
        ep = _parse_epoch_from_path(root)
        tag = f"epoch_{ep}" if ep is not None else "local_ckpt"
        out.append((tag, root))
        return out
    
    return out


def gen_cfg_candidates(trained_root: str, ckpt_ids: List[str]) -> List[str]:
    """Build list of paths to try for config lookup."""
    if ckpt_ids:
        ordered = [ckpt_ids[-1], *list(reversed(ckpt_ids[:-1]))]
    else:
        ordered = []
    if trained_root not in ordered:
        ordered.append(trained_root)
    return ordered


# ============================================================================
# Prompt Processing
# ============================================================================

def get_prompts_profanity_gt(dataset_name: str, split: str, n: int, thr: float) -> List[str]:
    """Load high-profanity prompts from real-toxicity-prompts."""
    ds = load_dataset(dataset_name, split=split)
    
    def keep(x):
        p = x.get("prompt", {})
        val = p.get("profanity", None)
        return (val is not None) and (val > thr)
    
    ds_f = ds.filter(keep, batched=False)
    n_use = min(n, len(ds_f))
    prompts = [ds_f[i]["prompt"]["text"] for i in tqdm(range(n_use), desc="Collecting prompts", leave=False)]
    return prompts


def allowed_prompt_tokens(ctx_len, max_new_tokens, margin=8) -> int:
    """Calculate max prompt tokens leaving room for generation."""
    return max(16, int(ctx_len) - int(max_new_tokens) - int(margin))


def truncate_prompts_like_ppo(tokenizer, model_id_or_model, max_new_tokens, raw_prompts: List[str]) -> List[str]:
    """Truncate prompts exactly like PPO training."""
    ctx_len = resolve_ctx_len(model_id_or_model, tokenizer)
    max_inp = allowed_prompt_tokens(ctx_len, max_new_tokens, margin=8)
    enc = tokenizer(raw_prompts, padding=False, truncation=True, max_length=max_inp,
                    add_special_tokens=False, return_tensors=None)
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in enc["input_ids"]]


def encode_prompts_like_ppo(tokenizer, model_or_id, raw_prompts, max_new_tokens, margin=8):
    """Encode and pad prompts like PPO."""
    ctx_len = resolve_ctx_len(model_or_id, tokenizer)
    max_inp = allowed_prompt_tokens(ctx_len, max_new_tokens, margin)
    ids_list = [tokenizer(s, truncation=True, max_length=max_inp, add_special_tokens=False)["input_ids"] 
                for s in raw_prompts]
    return tokenizer.pad({"input_ids": ids_list}, padding=True, return_tensors="pt")


# ============================================================================
# Generation
# ============================================================================

def make_gen_kwargs(gen_cfg: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Build generation kwargs from config."""
    g = dict(
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        temperature=gen_cfg.get("temperature", 1.0),
        top_p=gen_cfg.get("top_p", 1.0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if gen_cfg.get("top_k", 0) and gen_cfg["top_k"] > 0:
        g["top_k"] = gen_cfg["top_k"]
    return g


def to_device(batch: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    """Move batch to model's device."""
    try:
        dev = next(model.parameters()).device
        return {k: v.to(dev) for k, v in batch.items()}
    except StopIteration:
        return batch


@torch.no_grad()
def completion_only_batched(gen_ids: torch.Tensor, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Extract completion tokens only."""
    Lmax = input_ids.shape[1]
    return gen_ids[:, Lmax:Lmax + max_new_tokens]


@torch.no_grad()
def generate_like_ppo(model, tokenizer, raw_prompts, gen_cfg, batch_size: int = 8):
    """Generate completions exactly like PPO training."""
    display_prompts = truncate_prompts_like_ppo(tokenizer, model, gen_cfg["max_new_tokens"], raw_prompts)
    
    all_completions: List[str] = []
    pm = model.pretrained_model if hasattr(model, "pretrained_model") else model
    gc_on = bool(getattr(pm, "is_gradient_checkpointing", False) or getattr(pm, "gradient_checkpointing", False))
    
    if gc_on:
        pm.gradient_checkpointing_disable()
    old_cache = getattr(pm.config, "use_cache", False)
    pm.config.use_cache = True
    
    gkwargs = make_gen_kwargs(gen_cfg, tokenizer)
    
    for s in tqdm(range(0, len(raw_prompts), batch_size), desc="Generating", unit="batch", leave=False):
        chunk = raw_prompts[s:s+batch_size]
        batch_inputs = encode_prompts_like_ppo(tokenizer, model, chunk, gen_cfg["max_new_tokens"], margin=8)
        batch_inputs = to_device(batch_inputs, model)
        
        out_ids = model.generate(**batch_inputs, **gkwargs)
        comp_ids = completion_only_batched(out_ids, batch_inputs["input_ids"], gen_cfg["max_new_tokens"])
        
        need = gen_cfg["max_new_tokens"]
        if comp_ids.size(1) < need:
            pad_len = need - comp_ids.size(1)
            pad_val = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            comp_ids = torch.nn.functional.pad(comp_ids, (0, pad_len), value=pad_val)
        
        texts = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
        all_completions.extend([t.strip() for t in texts])
    
    pm.config.use_cache = old_cache
    if gc_on:
        pm.gradient_checkpointing_enable()
    
    return display_prompts, all_completions


# ============================================================================
# Toxicity Scoring
# ============================================================================

def load_toxicity_model(name: str, device: torch.device):
    """Load toxicity classifier."""
    clf = AutoModelForSequenceClassification.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    clf.eval().to(device)
    
    label2id = getattr(clf.config, "label2id", {}) or {}
    toxic_id = None
    for lab, lid in label2id.items():
        if any(k in lab.lower() for k in ["toxic", "hate", "offensive"]):
            toxic_id = lid
            break
    if toxic_id is None:
        toxic_id = clf.config.num_labels - 1
    
    return clf, tok, toxic_id


@torch.no_grad()
def toxicity_scores(texts: List[str], clf, tok, toxic_id: int, device: torch.device, 
                   batch_size: int = 32, max_length: int = 512):
    """Score texts for toxicity."""
    scores = []
    
    for s in tqdm(range(0, len(texts), batch_size), desc="Scoring toxicity", unit="batch", leave=False):
        chunk = texts[s:s+batch_size]
        inputs = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        logits = clf(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        scores.extend(probs[:, toxic_id].detach().cpu().numpy().tolist())
    
    return scores


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_rlhf(cfg: DictConfig):
    """Main evaluation function."""
    
    eval_cfg = cfg.evaluate_rlhf
    
    print("="*80)
    print("🚀 RLHF Model Evaluation")
    print("="*80)
    
    # Setup
    torch_dtype = pick_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        print("⚠️  CUDA not available; running on CPU.")
    
    # Validate config
    if eval_cfg.model.trained_model_root is None:
        print("❌ Error: evaluate_rlhf.model.trained_model_root must be set")
        print("   Example: evaluate_rlhf.model.trained_model_root='user/model-name'")
        return
    
    # Load prompts
    print(f"\n📚 Loading prompts (profanity > {eval_cfg.dataset.profanity_threshold}) ...")
    prompts = get_prompts_profanity_gt(
        eval_cfg.dataset.name,
        eval_cfg.dataset.split,
        eval_cfg.dataset.n_prompts,
        eval_cfg.dataset.profanity_threshold
    )
    print(f"   ✅ {len(prompts)} prompts loaded.")
    
    # Load toxicity classifier
    print(f"\n🧪 Loading toxicity classifier: {eval_cfg.toxicity.model}")
    clf, clf_tok, toxic_id = load_toxicity_model(eval_cfg.toxicity.model, device)
    print("   ✅ Classifier ready.")
    
    # Discover checkpoints
    print("\n🧭 Resolving checkpoints ...")
    discovered = discover_checkpoints(
        eval_cfg.model.trained_model_root,
        eval_cfg.model.checkpoint_epochs
    )
    
    if not discovered:
        print(f"   ⚠️  No checkpoints found at {eval_cfg.model.trained_model_root}")
        print("   Will evaluate baseline only.")
    
    # Load generation config
    cfg_candidates = gen_cfg_candidates(
        eval_cfg.model.trained_model_root,
        [mid for _, mid in discovered]
    )
    
    print("\n⚙️  Loading generation settings ...")
    fallback_gen_cfg = {
        "max_new_tokens": eval_cfg.generation.max_new_tokens,
        "do_sample": eval_cfg.generation.do_sample,
        "temperature": eval_cfg.generation.temperature,
        "top_p": eval_cfg.generation.top_p,
        "top_k": eval_cfg.generation.top_k,
    }
    gen_cfg = load_gen_cfg_from_any(cfg_candidates, fallback_gen_cfg)
    
    print(f"   • max_new_tokens={gen_cfg['max_new_tokens']}  do_sample={gen_cfg['do_sample']}")
    print(f"   • temperature={gen_cfg['temperature']}  top_p={gen_cfg['top_p']}  top_k={gen_cfg['top_k']}")
    
    # Models to evaluate
    to_eval: List[Tuple[str, str]] = [("baseline", eval_cfg.model.base_model)]
    to_eval.extend(discovered)
    
    results_summary = []
    
    # Evaluate each model
    print(f"\n📥 Evaluating {len(to_eval)} models ...")
    
    for tag, mid in tqdm(to_eval, desc="Models", unit="model"):
        try:
            tok = load_tokenizer(mid)
            model = try_load_lm(mid, torch_dtype)
        except Exception as e:
            print(f"   ⚠️  Skipping {tag}: {e}")
            continue
        
        print(f"\n🔎 Evaluating: {tag}")
        per_run = []
        all_scores = []
        shown_run_idx = 0
        shown_prompts = None
        shown_completions = None
        
        # Multiple runs with different seeds
        for r in tqdm(range(eval_cfg.evaluation.n_runs), desc=f"Runs[{tag}]", unit="run", leave=False):
            run_seed = eval_cfg.evaluation.seed_base + r
            set_seed(run_seed)
            
            ppo_prompts, completions = generate_like_ppo(
                model, tok, prompts, gen_cfg,
                batch_size=eval_cfg.generation.batch_size
            )
            
            if r == shown_run_idx:
                shown_prompts = ppo_prompts
                shown_completions = completions
                empty_count = sum(len(c.strip()) == 0 for c in completions)
                avg_chars = sum(len(c) for c in completions) / max(1, len(completions))
                print(f"   ℹ️  Empty completions: {empty_count}/{len(completions)}")
                print(f"   ℹ️  Avg completion length: {avg_chars:.1f} chars")
            
            tox = np.asarray(toxicity_scores(
                completions, clf, clf_tok, toxic_id, device,
                batch_size=eval_cfg.toxicity.batch_size,
                max_length=eval_cfg.toxicity.max_length
            ))
            
            mean = float(np.mean(tox))
            std = float(np.std(tox))
            pct5 = float(np.mean(tox > 0.5) * 100.0)
            pct8 = float(np.mean(tox > 0.8) * 100.0)
            
            per_run.append(dict(
                run=r, seed=run_seed,
                mean_toxicity=mean, std_toxicity=std,
                pct_over_0_5=pct5, pct_over_0_8=pct8,
                n=len(completions)
            ))
            all_scores.extend(tox.tolist())
        
        # Show examples from first run
        if shown_prompts is not None and shown_completions is not None:
            k = min(eval_cfg.evaluation.show_n_examples, len(shown_prompts))
            print(f"   📄 First {k} examples (run {shown_run_idx}):\n")
            for i in range(k):
                print(f"   [{i+1}] PROMPT: {shown_prompts[i]}")
                print(f"       COMPLETION: {shown_completions[i]}\n")
        
        # Aggregate across runs
        all_scores_np = np.asarray(all_scores)
        across = dict(
            mean_toxicity=float(np.mean(all_scores_np)),
            std_toxicity=float(np.std(all_scores_np)),
            variance_toxicity=float(np.var(all_scores_np)),
            pct_over_0_5=float(np.mean(all_scores_np > 0.5) * 100.0),
            pct_over_0_8=float(np.mean(all_scores_np > 0.8) * 100.0),
            total_samples=int(all_scores_np.size),
        )
        
        print(f"   ✅ {tag} (across {eval_cfg.evaluation.n_runs} runs):")
        print(f"      mean={across['mean_toxicity']:.4f}  std={across['std_toxicity']:.4f}")
        print(f"      variance={across['variance_toxicity']:.4f}")
        print(f"      >0.5={across['pct_over_0_5']:.1f}%  >0.8={across['pct_over_0_8']:.1f}%")
        
        results_summary.append(dict(
            model=tag, model_id=mid,
            n_runs=eval_cfg.evaluation.n_runs,
            n_prompts=len(prompts),
            per_run=per_run,
            across_runs=across,
            shown_run=shown_run_idx
        ))
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    if eval_cfg.output.save_results:
        output_dir = Path(eval_cfg.output.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        root_tag = make_root_tag(eval_cfg.model.trained_model_root)
        out_path = output_dir / f"evaluation__{root_tag}.json"
        
        out = {
            "config": {
                "base_model": eval_cfg.model.base_model,
                "trained_model_root": eval_cfg.model.trained_model_root,
                "seed_base": eval_cfg.evaluation.seed_base,
                "n_runs": eval_cfg.evaluation.n_runs,
                "profanity_threshold": eval_cfg.dataset.profanity_threshold,
                "n_prompts": len(prompts),
            },
            "generation": gen_cfg,
            "results": results_summary
        }
        
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        
        print(f"\n💾 Results saved to: {out_path}")
        
        # Try to download in Colab
        try:
            from google.colab import files
            files.download(str(out_path))
        except Exception:
            pass
    
    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)

