"""
Utility functions for RLHF training.
"""

import os, sys
import random
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    set_seed,
    AutoConfig
)
from datasets import load_dataset
from omegaconf import OmegaConf

# Import from the IRL pipeline
from irl_pipeline.irl.reward_computer import SimplifiedIRLRewardComputer


class LengthSampler:
    """Samples a length within a specified range."""
    
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self) -> int:
        return random.randint(self.min_length, self.max_length)


def compute_allowed_prompt_len(tokenizer, model_name, gen_max_new_tokens, safety_margin=8):
    """Compute the maximum prompt length that leaves room for generation."""
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        ctx = int(getattr(cfg, "max_position_embeddings", tokenizer.model_max_length))
    except Exception:
        ctx = int(tokenizer.model_max_length)

    # Clamp to something sane if needed
    if ctx is None or ctx > 100_000_000:
        ctx = 4096  # safe default

    return max(16, ctx - int(gen_max_new_tokens) - int(safety_margin))


def build_dataset(config: Dict) -> Tuple[Any, Any, AutoTokenizer]:
    """Build train/test datasets and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset(config.dataset.name, split="train")

    # Filter by toxicity metric
    metric = getattr(config.dataset, "filter_metric", "toxicity")
    thr = float(getattr(config.dataset, "toxicity_threshold", 0.5))

    def filter_fn(sample):
        p = sample.get("prompt", {})
        m = p.get(metric, None)
        return (m is not None) and (m >= thr)

    ds = ds.filter(filter_fn, batched=False)

    # Compute allowed prompt length
    allowed_prompt_len = compute_allowed_prompt_len(
        tokenizer=tokenizer,
        model_name=config.model.name,
        gen_max_new_tokens=config.model.generation.max_new_tokens,
        safety_margin=8,
    )
    print("allowed_prompt_len:", allowed_prompt_len)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        enc = tokenizer(
            prompt,
            truncation=True,
            max_length=allowed_prompt_len,
            add_special_tokens=False,
        )
        sample["input_ids"] = enc["input_ids"]
        sample["query"] = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    ds = ds.train_test_split(test_size=config.dataset.test_size, seed=config.training.seed)
    return ds["train"], ds["test"], tokenizer


def completion_only(generated_ids: torch.LongTensor,
                    input_ids: torch.LongTensor,
                    max_new_tokens: int) -> torch.LongTensor:
    """Extract only the generated completion from full sequence."""
    prompt_len = input_ids.shape[1]
    return generated_ids[:, prompt_len : prompt_len + max_new_tokens]


def collator(data: List[Dict]) -> Dict:
    """Custom collator function for PPO training."""
    return {key: [d[key] for d in data] for key in data[0]}


def setup_wandb(config) -> Any:
    """Initialize Weights & Biases logging."""
    try:
        import wandb
        from omegaconf import OmegaConf

        name = config.wandb.name
        if not name:
            model_name = (config.model.name or "unknown-model").split("/")[-1]
            name = f"{model_name}-{config.now}"
        
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=name,
            config=OmegaConf.to_container(config, resolve=True),
            reinit=True,
        )
        config.wandb.name = name
        return run
    except ImportError:
        print("wandb not installed. Skipping wandb initialization.")
        return None
    except Exception as e:
        print(f"Error initializing wandb: {str(e)}")
        return None


def load_reward_model(model_id: str, device: str) -> Tuple[Any, Any]:
    """Load reward model for RLHF training."""
    
    if "roberta" in model_id.lower():
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_id)
        model = RobertaForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Setting pad_token to eos_token ({tokenizer.eos_token})")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            if hasattr(model.config, 'num_labels') and model.config.num_labels == 1:
                print(f"Loaded model with single label head")
            elif not hasattr(model, 'score'):
                print(f"Model doesn't have a score attribute, will use logits directly")
                
        except Exception as e:
            print(f"Error loading as sequence classification model: {e}")
            print("Trying to load as a causal language model with value head...")
            
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            class RewardModelWrapper(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    self.has_v_head = hasattr(base_model, 'v_head')
                    
                def forward(self, **inputs):
                    outputs = self.base_model(**inputs, output_hidden_states=True)
                    
                    if self.has_v_head:
                        hidden_states = outputs.hidden_states[-1]
                        
                        if 'attention_mask' in inputs:
                            attention_mask = inputs['attention_mask']
                            expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            masked_hidden = hidden_states * expanded_mask
                            sum_hidden = torch.sum(masked_hidden, dim=1)
                            token_count = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1.0)
                            pooled_hidden = sum_hidden / token_count
                        else:
                            batch_size = hidden_states.size(0)
                            last_token_indices = torch.tensor([hidden_states.size(1)-1] * batch_size, 
                                                             device=hidden_states.device)
                            batch_indices = torch.arange(batch_size, device=hidden_states.device)
                            pooled_hidden = hidden_states[batch_indices, last_token_indices]
                        
                        values = self.base_model.v_head(pooled_hidden)
                        
                        class SimpleOutput:
                            def __init__(self, logits):
                                self.logits = logits
                        
                        return SimpleOutput(torch.cat([-values, values], dim=1))
                    else:
                        return outputs
            
            model = RewardModelWrapper(base_model).to(device)
            print("Successfully loaded custom reward model with wrapper")
        
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Setting model's pad_token_id to {model.config.pad_token_id}")
    
    model.eval()
    print(f"Reward model loaded and set to evaluation mode")
    
    return model, tokenizer


def evaluate_toxicity(
    model, 
    ppo_trainer, 
    tokenizer, 
    reward_model, 
    reward_tokenizer, 
    dataset, 
    config, 
    epoch
) -> Tuple[float, List[Dict]]:
    """Evaluate model toxicity on a dataset."""
    
    output_dir = os.path.join(os.getcwd(), f"outputs/{config.now}")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    device = ppo_trainer.accelerator.device
    
    eval_size = min(100, len(dataset))
    eval_indices = random.sample(range(len(dataset)), eval_size)
    eval_samples = [dataset[i] for i in eval_indices]
    
    gen_kwargs = {
        "max_new_tokens": config.model.generation.max_new_tokens,
        "do_sample": config.model.generation.do_sample,
        "temperature": 1.0,
        "top_p": config.model.generation.top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
        
    toxicity_scores = []
    generations = []
    
    B = int(getattr(config.model.generation, "eval_batch_size", 16))
    for i in range(0, len(eval_samples), B):
        chunk = eval_samples[i:i+B]
        queries = [s["query"] for s in chunk]

        toks = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
        resp = ppo_trainer.model.generate(**toks, **gen_kwargs)
        comp_ids = completion_only(resp, toks.input_ids, config.model.generation.max_new_tokens)

        texts = [tokenizer.decode(comp_ids[j], skip_special_tokens=True) for j in range(comp_ids.size(0))]

        inputs = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            try:
                outputs = reward_model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits.float()
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        p_tox = torch.sigmoid(logits.squeeze(-1))
                    else:
                        tox_idx, _ = _tox_non_tox_indices(reward_model)
                        p_tox = torch.softmax(logits, dim=1)[:, tox_idx]
                else:
                    val = outputs[0].float().squeeze(-1)
                    p_tox = torch.sigmoid(val)
                chunk_scores = p_tox.tolist()
            except Exception:
                chunk_scores = [0.5] * len(texts)

        for q, r, t in zip(queries, texts, chunk_scores):
            toxicity_scores.append(float(t))
            generations.append({"query": q, "response": r, "toxicity": float(t)})

    avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)
    
    output_file = os.path.join(eval_dir, f"generations_epoch_{epoch}.json")
    with open(output_file, "w") as f:
        json.dump(generations, f, indent=2)
    
    plt.figure(figsize=(10, 6))
    plt.hist(toxicity_scores, bins=20, alpha=0.7)
    plt.xlabel("Toxicity Score")
    plt.ylabel("Frequency")
    plt.title(f"Toxicity Distribution (Epoch {epoch})")
    plt.axvline(avg_toxicity, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {avg_toxicity:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(eval_dir, f"toxicity_dist_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()
    
    return avg_toxicity, generations


def safe_reward_computation(reward_model, toxicity_inputs, device):
    """Safely compute rewards with error handling."""
    with torch.no_grad():
        try:
            outputs = reward_model(**toxicity_inputs)
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits.float()
                if logits.ndim == 2 and logits.shape[1] == 1:
                    raw_values = -logits.squeeze(-1)
                else:
                    tox_idx, _ = _tox_non_tox_indices(reward_model)
                    raw_values = -logits[:, tox_idx]
            else:
                val = outputs[0].float().squeeze(-1)
                raw_values = -val

            if torch.isnan(raw_values).any() or torch.isinf(raw_values).any():
                print("Warning: NaN/Inf in reward output; replacing with zeros")
                raw_values = torch.where(
                    torch.isnan(raw_values) | torch.isinf(raw_values),
                    torch.zeros_like(raw_values),
                    raw_values,
                )
            return raw_values
            
        except Exception as e:
            print(f"Error in reward computation: {e}")
            bs = toxicity_inputs['input_ids'].size(0)
            return torch.zeros(bs, device=device)


def _tox_non_tox_indices(model):
    """Get toxic/non-toxic indices from model config."""
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        id2label = {int(k): str(v).lower() for k, v in id2label.items()}
        tox_idx = next((i for i,lab in id2label.items() if "toxic" in lab), None)
    else:
        tox_idx = None
    
    if tox_idx is None:
        tox_idx = 1  # common: 0=NON_TOXIC, 1=TOXIC
    non_idx = 1 - tox_idx if tox_idx in (0,1) else None
    return tox_idx, non_idx


# IRL reward integration
class IRLReward:
    """IRL posterior-based reward computer."""
    
    def __init__(
        self,
        posterior_dir: str,
        global_norm_dir: str,
        base_model_name: str,
        device: torch.device,
        features_on_cpu: bool = False,
        feature_max_length: int = 256,
        feature_batch_size: int = 16,
        use_platt: bool = False,
        platt_a: float = 1.0,
        platt_b: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device("cpu") if features_on_cpu else device
        self.mu = torch.from_numpy(np.load(os.path.join(posterior_dir, "mu.npy"))).to(device=device, dtype=dtype)
        self.sig = torch.from_numpy(np.load(os.path.join(posterior_dir, "sig.npy"))).to(device=device, dtype=dtype)
        self.mean = torch.from_numpy(np.load(os.path.join(global_norm_dir, "global_mean.npy"))).to(device=device, dtype=dtype)
        self.std  = torch.from_numpy(np.load(os.path.join(global_norm_dir, "global_std.npy"))).to(device=device, dtype=dtype).clamp_min_(1e-6)

        d_mu = self.mu.numel()
        print(f"🔍 [DEBUG] IRLReward Init:")
        print(f"  - base_model_name: {base_model_name}")
        print(f"  - feature_max_length: {feature_max_length}")
        print(f"  - mu.shape: {self.mu.shape}")
        print(f"  - sig.shape: {self.sig.shape}")
        print(f"  - mean.shape: {self.mean.shape}")
        print(f"  - std.shape: {self.std.shape}")
        print(f"  - d_mu (theta dimension): {d_mu}")
        
        whitening_stats_disabled = {"disable_whitening": True}
        device_feat = "cpu" if features_on_cpu else str(device)
        self._feat = SimplifiedIRLRewardComputer(
            artifact_name=None,
            base_model_name=base_model_name,
            likelihood_type="bradley_terry",
            normalization_strategy="none",
            n_posterior_samples=1,
            device=device_feat,
            theta_samples=torch.randn(1, d_mu, device=device, dtype=dtype),
            whitening_stats=whitening_stats_disabled,
        )
        print(f"  - IRL features device: {device_feat} | IRL math device: {self.device}")
        self._max_len = feature_max_length
        self._batch = feature_batch_size

        self.use_platt = use_platt
        self.platt_a = float(platt_a)
        self.platt_b = float(platt_b)
        
        self._has_norm = False
        self.theta = None
        self.theta_sig = None
        self.max_length = feature_max_length
        self.batch_size = feature_batch_size
        self.n_samples = 100

    @torch.no_grad()
    def score_texts(self, texts, sample_theta: bool = False, max_length: int | None = None):
        """Score texts with IRL reward."""
        phi = self.extract_features(texts, max_length=max_length or self.max_length, batch_size=self.batch_size)

        if not getattr(self, "_has_norm", False):
            raise RuntimeError("IRL scorer: normalization not set. Call set_normalization(...) first.")
        phi_normalized = (phi - self.mean) / self.std

        n = int(getattr(self, "n_samples", 1)) if (sample_theta and getattr(self, "theta_sig", None) is not None) else 1
        if n <= 1:
            theta = self.theta
            R = (theta @ phi_normalized.T).flatten()
        else:
            theta0 = self.theta.unsqueeze(0)
            sig0   = self.theta_sig.unsqueeze(0)
            theta_samples = theta0 + torch.randn(
                n, *theta0.shape[1:],
                device=theta0.device, dtype=theta0.dtype
            ) * sig0
            R = (theta_samples @ phi_normalized.T).mean(dim=0)

        if self.use_platt:
            result = torch.sigmoid(self.platt_a * R + self.platt_b)
            return result
        return R

    def extract_features(self, texts, max_length: int, batch_size: int):
        return self._feat.extract_features(texts, max_length=max_length, batch_size=batch_size)

    def set_normalization(self, mean, std):
        self.mean = torch.from_numpy(mean).to(device=self.device, dtype=torch.float32)
        self.std = torch.from_numpy(std).to(device=self.device, dtype=torch.float32).clamp_min_(1e-6)
        self._has_norm = True

    def set_theta_from_mu(self, mu, sig):
        self.theta = torch.from_numpy(mu).to(device=self.device, dtype=torch.float32)
        self.theta_sig = torch.from_numpy(sig).to(device=self.device, dtype=torch.float32)

    def set_platt_single(self, a, b):
        self.platt_a = float(a)
        self.platt_b = float(b)


def load_irl_reward(config: Dict, device: torch.device):
    """Load IRL reward from config."""
    irl_cfg = config.irl
    base_model_name = getattr(irl_cfg, "base_model_name", None) or config.model.name
    
    print(f"🔍 [DEBUG] IRL Reward Setup:")
    print(f"  - base_model_name: {base_model_name}")
    print(f"  - feature_max_length: {getattr(irl_cfg, 'feature_max_length', 256)}")
    print(f"  - posterior_dir: {irl_cfg.posterior_dir}")
    print(f"  - global_norm_dir: {irl_cfg.global_norm_dir}")

    reward = IRLReward(
        posterior_dir=irl_cfg.posterior_dir,
        global_norm_dir=irl_cfg.global_norm_dir,
        base_model_name=base_model_name,
        device=device,
        features_on_cpu=getattr(irl_cfg, "features_on_cpu", True),
        feature_max_length=getattr(irl_cfg, "feature_max_length", 256),
        feature_batch_size=getattr(irl_cfg, "feature_batch_size", 16),
        use_platt=getattr(irl_cfg, "use_platt", False),
        platt_a=getattr(irl_cfg, "platt_a", 1.0),
        platt_b=getattr(irl_cfg, "platt_b", 0.0),
        dtype=torch.float32,
    )

    reward.n_samples = int(getattr(irl_cfg, "n_samples", 100))
    return reward, None


def attach_vi_artifacts(irl_reward, vi_dir: str, use_round: int | None = None):
    """Attach VI artifacts (normalization, posterior, Platt params) to IRL reward."""
    vi_dir = Path(vi_dir)
    print(f"🔍 [DEBUG] Attaching VI Artifacts:")
    print(f"  - vi_dir: {vi_dir}")
    print(f"  - use_round: {use_round}")
    
    mean = np.load(vi_dir / "global_mean.npy")
    std  = np.load(vi_dir / "global_std.npy")
    print(f"  - global_mean.shape: {mean.shape}")
    print(f"  - global_std.shape: {std.shape}")
    
    rounds = sorted([p for p in vi_dir.glob("round_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[1]))
    if not rounds:
        raise FileNotFoundError(f"No rounds found in {vi_dir}")
    rd = rounds[-1] if use_round is None else vi_dir / f"round_{use_round}"
    print(f"  - Selected round: {rd}")

    mu  = np.load(rd / "mu.npy")
    sig = np.load(rd / "sig.npy")
    print(f"  - mu.shape: {mu.shape}")
    print(f"  - sig.shape: {sig.shape}")
    
    with open(vi_dir / "summary.json", "r") as f:
        summ = json.load(f)
    
    r_idx = int(rd.name.split("_")[1]) - 1
    a = float(summ[r_idx]["platt_single"]["a"])
    b = float(summ[r_idx]["platt_single"]["b"])
    print(f"  - Platt calibration: a={a:.6f}, b={b:.6f}")
    if abs(a) > 1e-8:
        print(f"  - R* = -b/a = {-b/a:.6f}")
    else:
        print(f"  - R* = undefined (a ≈ 0, using midpoint threshold instead)")

    irl_reward.set_normalization(mean, std)
    irl_reward.set_theta_from_mu(mu, sig)
    irl_reward.set_platt_single(a, b)
    
    if abs(a) > 1e-8:
        irl_reward.R_star = -b / a
    else:
        if 'R_star_mid' in summ[0]:
            irl_reward.R_star = float(summ[0]['R_star_mid'])
        else:
            irl_reward.R_star = 0.0
        print(f"  - Using fallback R* = {irl_reward.R_star:.6f}")
    print(f"  - VI artifacts attached successfully!")



def get_model_safe_name(model_name: str) -> str:
    """Convert model name to safe directory name (same as IRL training)."""
    return model_name.replace("/", "_").replace("-", "_")


def verify_irl_compat(irl_reward, expected_base: str, sample_text: str = "Test sample."):
    """
    Sanity checks that the IRL posterior matches the feature extractor:
      - base model name (best-effort print)
      - feature dimension equals posterior dimension
    Also prints a short "probe" extraction summary.
    """
    try:
        base_used = getattr(irl_reward._feat, "base_model_name", None)
    except Exception:
        base_used = None

    print("[IRL] Verifying compatibility...")
    print(f"  expected base      : {expected_base}")
    print(f"  feature extractor  : {base_used}")

    # Extract a tiny feature vector and compare dims
    phi = irl_reward.extract_features([sample_text], max_length=irl_reward.max_length, batch_size=1)
    d_phi = int(phi.shape[-1])
    d_theta = int(irl_reward.mu.numel())
    print(f"  feature dim        : {d_phi}")
    print(f"  posterior dim      : {d_theta}")

    if d_phi != d_theta:
        raise ValueError(
            f"[IRL] Dimension mismatch: feature dim {d_phi} != posterior dim {d_theta}. "
            f"Did you train VI with a different base model/tokenization/max_length?"
        )

    # If available, gentle warning for base mismatch
    if base_used and (base_used != expected_base):
        print(f"  WARNING: base mismatch. Posterior was trained with '{base_used}', "
              f"but RLHF config indicates '{expected_base}'. "
              f"This MAY still work if features are identical, but be careful.")

    if not getattr(irl_reward, "_has_norm", False):
        raise RuntimeError("[IRL] Normalization not set. attach_vi_artifacts must run first.")

    print("[IRL] Compatibility check: OK.\n")


