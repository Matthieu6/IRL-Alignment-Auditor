"""
Main RLHF training script for detoxifying language models.
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from datetime import datetime
from torch.optim import Adam
from tqdm import tqdm

# Set environment variables to disable TorchDynamo completely
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_TORCH_COMPILE'] = '1'
os.environ['TORCH_LOGS'] = 'off'
os.environ['TORCHDYNAMO_VERBOSE'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable TorchDynamo compilation globally
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Lightweight CUDA speed-ups
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from huggingface_hub import HfApi

from irl_pipeline.rlhf.utilities import (
    build_dataset,
    collator,
    setup_wandb,
    load_reward_model,
    evaluate_toxicity,
    LengthSampler,
    safe_reward_computation,
    load_irl_reward,
    attach_vi_artifacts,
    verify_irl_compat,
    get_model_safe_name,
    completion_only
)


def _resolve_irl_paths(cfg):
    """Resolve and validate IRL paths; auto-detect model dir from irl_root + model name."""
    irl = cfg.irl
    irl_root = getattr(irl, "irl_root", None)
    gdir = getattr(irl, "global_norm_dir", None)
    pdir = getattr(irl, "posterior_dir", None)
    urnd = getattr(irl, "use_round", None)
    
    # Auto-resolve from irl_root + model name + round
    if irl_root and not gdir:
        base_model = getattr(irl, "base_model_name", None) or cfg.model.name
        model_safe = get_model_safe_name(base_model)
        gdir = os.path.join(irl_root, model_safe)
    
    # Auto-resolve posterior_dir from global_norm_dir + round
    if gdir and urnd is not None:
        auto_pdir = os.path.join(gdir, f"round_{int(urnd)}")
        if pdir is None:
            pdir = auto_pdir
        else:
            norm_pdir = os.path.normpath(pdir)
            norm_auto = os.path.normpath(auto_pdir)
            if norm_pdir != norm_auto:
                raise ValueError(
                    f"[IRL] Mismatch: posterior_dir={norm_pdir} but global_norm_dir/use_round -> {norm_auto}"
                )
    
    # Existence checks
    if gdir and not os.path.exists(gdir):
        raise FileNotFoundError(f"[IRL] global_norm_dir not found: {gdir}")
    if pdir and not os.path.exists(pdir):
        raise FileNotFoundError(f"[IRL] posterior_dir not found: {pdir}")
    
    # Update config with resolved paths
    from omegaconf import open_dict
    with open_dict(irl):
        irl.global_norm_dir = gdir
        irl.posterior_dir = pdir
        irl.use_round = urnd
    
    return gdir, pdir, urnd


def train_rlhf(cfg: DictConfig) -> float:
    """Main training function."""
    
    # Add current timestamp
    with open_dict(cfg):
        cfg.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    fast_start = bool(getattr(cfg.training, "fast_start", True))

    with open_dict(cfg):
        # If the selected preset was loaded under the 'model' group, flatten it
        if isinstance(cfg.model, DictConfig) and "model" in cfg.model:
            cfg.model = cfg.model.model
        
        # Pull up missing top-level keys from model preset
        for k in ("training", "dataset", "output", "wandb"):
            if hasattr(cfg.model, k) and not hasattr(cfg, k):
                setattr(cfg, k, getattr(cfg.model, k))

    # Print configuration
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directories
    output_dir = os.path.join(os.getcwd(), f"outputs/{cfg.now}")
    os.makedirs(output_dir, exist_ok=True)
    eval_dir = os.path.join(output_dir, "evaluation")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Setup WandB logging
    wandb_run = setup_wandb(cfg)
    
    # Build dataset and tokenizer
    print("Building dataset...")
    train_dataset, test_dataset, tokenizer = build_dataset(cfg)
    tokenizer.padding_side = "left"

    print(f"Train set: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")

    # Prepare model loading kwargs
    model_kwargs = {}
    if hasattr(cfg.model, 'attn_implementation'):
        model_kwargs['attn_implementation'] = cfg.model.attn_implementation
    if hasattr(cfg.model, 'use_cache'):
        model_kwargs['use_cache'] = cfg.model.use_cache

    # Load model and add value head
    print(f"Loading model {cfg.model.name}...")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    _dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        **model_kwargs,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Memory-friendly training
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
    else:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Ensure model sees the same pad id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "pretrained_model") and getattr(model.pretrained_model.config, "pad_token_id", None) is None:
        model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id

    # Create reference model
    ref_model = create_reference_model(model)
    
    # Create optimizer
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.model.learning_rate
    )
    
    # Create learning rate scheduler
    total_steps = cfg.training.num_train_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Get PPO parameters from RLHF config
    ppo_params = {
        "model_name": cfg.model.name,
        "learning_rate": cfg.model.learning_rate,
        "log_with": "wandb" if wandb_run else None,
    }

    if "target" in cfg.model:
        ppo_params["target_kl"] = cfg.model.target

    for k in ("ppo_epochs", "init_kl_coef", "cliprange", "cliprange_value",
              "vf_coef", "adap_kl_ctrl", "use_score_norm", "ratio_threshold"):
        if k in cfg.model:
            ppo_params[k] = cfg.model[k]
    
    # Handle batch size parameters
    batch_size = cfg.model.batch_size
    mini_batch_size = cfg.model.mini_batch_size
    gradient_accumulation_steps = cfg.model.gradient_accumulation_steps

    if batch_size % (mini_batch_size * gradient_accumulation_steps) != 0:
        if batch_size >= gradient_accumulation_steps:
            new_mini_batch_size = batch_size // gradient_accumulation_steps
            print(f"Warning: Adjusting mini_batch_size from {mini_batch_size} to {new_mini_batch_size}")
            mini_batch_size = new_mini_batch_size
        else:
            new_gradient_accumulation_steps = 1
            new_mini_batch_size = batch_size
            print(f"Warning: Adjusting gradient_accumulation_steps and mini_batch_size")
            gradient_accumulation_steps = new_gradient_accumulation_steps
            mini_batch_size = new_mini_batch_size

    ppo_params["batch_size"] = batch_size
    ppo_params["mini_batch_size"] = mini_batch_size
    ppo_params["gradient_accumulation_steps"] = gradient_accumulation_steps

    # Create PPO config
    ppo_config = PPOConfig(**ppo_params)
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    
    # Load toxicity model
    print(f"Loading toxicity model {cfg.model.reward_model}...")
    reward_model, reward_tokenizer = load_reward_model(
        cfg.model.reward_model,
        ppo_trainer.accelerator.device
    )

    # Load IRL reward if configured
    use_irl_reward = hasattr(cfg, "irl") and (
        getattr(cfg.irl, "irl_root", None) or getattr(cfg.irl, "global_norm_dir", None)
    )
    irl_reward = None
    if use_irl_reward:
        gdir, pdir, urnd = _resolve_irl_paths(cfg)
        
        print("\n" + "="*72)
        print("[IRL] Configuration summary")
        print("="*72)
        print(f"  base_model_name    : {getattr(cfg.irl, 'base_model_name', None) or cfg.model.name}")
        print(f"  global_norm_dir    : {gdir}")
        print(f"  posterior_dir      : {pdir}")
        print(f"  use_round          : {urnd}")
        print(f"  features_on_cpu    : {getattr(cfg.irl, 'features_on_cpu', True)}")
        print(f"  n_samples (theta)  : {getattr(cfg.irl, 'n_samples', 100)}")
        print(f"  sample_theta_each  : {getattr(cfg.irl, 'sample_theta_each_step', True)}")
        print(f"  reward_scale/clip  : {getattr(cfg.irl, 'reward_scale', 20)} / {getattr(cfg.irl, 'reward_clip', 2)}")
        print("="*72 + "\n")
        
        irl_reward, _ = load_irl_reward(cfg, ppo_trainer.accelerator.device)
        
        attach_vi_artifacts(
            irl_reward,
            vi_dir=gdir,
            use_round=urnd
        )
        
        verify_irl_compat(
            irl_reward=irl_reward,
            expected_base=getattr(cfg.irl, "base_model_name", None) or cfg.model.name,
            sample_text="Hello world."
        )

    # Use IRL posterior as training reward if available
    sample_theta_each_step = False
    if use_irl_reward:
        sample_theta_each_step = getattr(cfg.irl, "sample_theta_each_step", True)
    
    # Setup generation parameters
    output_length_sampler = LengthSampler(
        cfg.model.generation.output_min_length,
        cfg.model.generation.output_max_length
    )
    
    # Initial evaluation
    print("Performing initial evaluation...")
    initial_toxicity, _ = evaluate_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="initial"
    )
    
    print(f"Initial average toxicity: {initial_toxicity:.4f}")

    # Save evaluation results
    with open(os.path.join(eval_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Epoch 0: Average toxicity = {initial_toxicity:.4f}\n")
    
    # Log initial metrics
    if wandb_run:
        wandb_run.log({"eval/initial_toxicity": initial_toxicity})
    
    # Create reward stats tracker
    reward_stats = {
        'epoch': [],
        'raw_rewards_mean': [],
        'raw_rewards_std': [],
        'nan_inf_count': [],
    }
    
    # Training loop
    print("Starting training loop...")
    training_start_time = time.time()
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=cfg.training.num_train_epochs):
        if epoch >= cfg.training.num_train_epochs:
            break
        
        # Process batch
        query_tensors = batch["input_ids"]
        
        # Get response from policy model
        device = ppo_trainer.accelerator.device
        gen_len = cfg.model.generation.max_new_tokens
        generation_kwargs = {
            "max_new_tokens": gen_len,
            "do_sample": cfg.model.generation.do_sample,
            "temperature": 1.0,
            "top_p": cfg.model.generation.top_p,
            "pad_token_id": tokenizer.pad_token_id,
        }

        batch_inputs = tokenizer.pad(
            {"input_ids": [q.squeeze().tolist() for q in query_tensors]},
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Generate with gradient checkpointing disabled
        pm = ppo_trainer.model.pretrained_model if hasattr(ppo_trainer.model, "pretrained_model") else ppo_trainer.model
        gc_on = getattr(pm, "is_gradient_checkpointing", False) or getattr(pm, "gradient_checkpointing", False)
        if gc_on: pm.gradient_checkpointing_disable()
        old_cache = getattr(pm.config, "use_cache", False)
        pm.config.use_cache = True
        with torch.no_grad():
            responses_full = ppo_trainer.model.generate(**batch_inputs, **generation_kwargs)
        pm.config.use_cache = old_cache
        if gc_on: pm.gradient_checkpointing_enable()

        # Extract completion only
        comp_ids = completion_only(responses_full, batch_inputs["input_ids"], gen_len)

        # Ensure fixed length per sample for PPO
        eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        response_tensors = []
        for i in range(comp_ids.size(0)):
            resp_i = comp_ids[i]
            if resp_i.numel() < gen_len:
                pad = torch.full((gen_len - resp_i.numel(),), eos_id,
                                device=resp_i.device, dtype=resp_i.dtype)
                resp_i = torch.cat([resp_i, pad], dim=0)
            response_tensors.append(resp_i)

        batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Compute rewards
        texts = batch["response"]

        if use_irl_reward:
            # IRL reward: higher = less toxic
            raw_values = irl_reward.score_texts(texts, sample_theta=sample_theta_each_step)

            # Scale and clip for PPO stability
            scale = 1.0 / float(getattr(cfg.irl, "reward_scale", 20.0))
            clip  = float(getattr(cfg.irl, "reward_clip", 2.0))

            rv = torch.stack([v.detach().float() for v in raw_values])
            rv_scaled = torch.clamp(rv * scale, -clip, clip)
            rv_scaled = rv_scaled.to(ppo_trainer.accelerator.device)

            clip_rate = (rv.abs() * scale > clip).float().mean().item()
            try:
                ppo_trainer.accelerator.log({"rewards/clip_rate": clip_rate})
            except Exception:
                pass

            rewards = [rv_scaled[i] for i in range(rv_scaled.size(0))]
            raw_toxicity_labels = [float(x) for x in rv_scaled.tolist()]
        else:
            # Use classifier reward
            toxicity_inputs = reward_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(ppo_trainer.accelerator.device)

            raw_values = safe_reward_computation(
                reward_model, toxicity_inputs, ppo_trainer.accelerator.device
            )

            if cfg.model.use_raw_logits:
                raw_toxicity_labels = raw_values.tolist()
                raw_toxicity_labels = [
                    0.0 if (not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x)) else x
                    for x in raw_toxicity_labels
                ]
                rewards = [torch.tensor(output, device=ppo_trainer.accelerator.device) for output in raw_toxicity_labels]
            else:
                softmax_values = torch.nn.functional.softmax(raw_values.view(-1, 1), dim=1)[:, 0]
                raw_toxicity_labels = softmax_values.tolist()
                raw_toxicity_labels = [
                    0.0 if (not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x)) else x
                    for x in raw_toxicity_labels
                ]
                rewards = [torch.tensor(output, device=ppo_trainer.accelerator.device) for output in raw_toxicity_labels]

        # Calculate statistics for logging
        rewards_tensor = torch.as_tensor(
            [float(r.item() if torch.is_tensor(r) else r) for r in rewards],
            dtype=torch.float32
        )
        raw_mean = rewards_tensor.mean().item()
        raw_std  = rewards_tensor.std(unbiased=False).item()

        # Store statistics
        reward_stats['epoch'].append(epoch)
        reward_stats['raw_rewards_mean'].append(raw_mean)
        reward_stats['raw_rewards_std'].append(raw_std)
        
        # Count NaN/Inf values
        nan_inf_count = sum(1 for x in raw_toxicity_labels if not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x))
        reward_stats['nan_inf_count'].append(nan_inf_count)
        
        # Print reward stats periodically
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} reward stats:")
            print(f"  Rewards - Mean: {raw_mean:.4f}, Std: {raw_std:.4f}")
            print(f"  NaN/Inf values replaced: {nan_inf_count}/{len(raw_toxicity_labels)}")
        
        # Run PPO update
        stats = safe_ppo_step(ppo_trainer, query_tensors, response_tensors, rewards)
        
        # Augment stats with reward metrics
        stats["rewards/mean"] = raw_mean
        stats["rewards/std"] = raw_std
        stats["current_epoch"] = epoch
        
        # Log stats
        safe_log_stats(ppo_trainer, stats, batch, rewards)
        
        # Save model checkpoint
        if (epoch + 1) % cfg.training.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
            print(f"Saving model checkpoint to {checkpoint_path}")
            
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                with open(os.path.join(checkpoint_path, "rlhf_config.yaml"), "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                
                reward_df = pd.DataFrame(reward_stats)
                reward_df.to_csv(os.path.join(output_dir, "reward_stats.csv"), index=False)
        
        # Push checkpoint to Hub if enabled
        if (
            cfg.output.push_to_hub
            and cfg.output.push_checkpoints_to_hub
            and (epoch + 1) % cfg.output.checkpoint_push_freq == 0
            and ppo_trainer.accelerator.is_main_process
        ):
            try:
                if (epoch + 1) % cfg.training.save_freq != 0:
                    temp_checkpoint_path = os.path.join(checkpoint_dir, f"temp-checkpoint-epoch-{epoch+1}")
                    ppo_trainer.save_pretrained(temp_checkpoint_path)
                    tokenizer.save_pretrained(temp_checkpoint_path)
                    with open(os.path.join(temp_checkpoint_path, "rlhf_config.yaml"), "w") as f:
                        f.write(OmegaConf.to_yaml(cfg))
                    checkpoint_path = temp_checkpoint_path
                else:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}")

                api = HfApi()
                owner = cfg.output.hf_owner
                base_name = cfg.output.repository_name or f"{cfg.model.name.split('/')[-1]}-detox"
                checkpoint_repo_name = f"{base_name}-checkpoint-epoch-{epoch+1}"
                checkpoint_repo_id = f"{owner}/{checkpoint_repo_name}"

                print(f"Pushing checkpoint to Hugging Face Hub: {checkpoint_repo_id}")
                api.create_repo(
                    repo_id=checkpoint_repo_id,
                    private=getattr(cfg.output, "private", False),
                    exist_ok=True,
                    repo_type="model",
                )
                api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=checkpoint_repo_id,
                    commit_message=f"Checkpoint epoch {epoch+1}",
                )
                print(f"Successfully pushed checkpoint to {checkpoint_repo_id}")

                if (epoch + 1) % cfg.training.save_freq != 0 and os.path.exists(temp_checkpoint_path):
                    import shutil
                    shutil.rmtree(temp_checkpoint_path)

            except Exception as e:
                print(f"Error pushing checkpoint to Hugging Face Hub: {str(e)}")
        
        # Run evaluation
        if (epoch + 1) % cfg.training.eval_freq == 0:
            print(f"\nEvaluating at epoch {epoch+1}...")
            
            avg_toxicity, _ = evaluate_toxicity(
                model=model,
                ppo_trainer=ppo_trainer,
                tokenizer=tokenizer,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                dataset=test_dataset,
                config=cfg,
                epoch=epoch+1
            )
            
            print(f"Epoch {epoch+1}: Average toxicity = {avg_toxicity:.4f}")

            # Save evaluation results
            with open(os.path.join(eval_dir, "evaluation_results.txt"), "a") as f:
                f.write(f"Epoch {epoch+1}: Average toxicity = {avg_toxicity:.4f}\n")
            
            # Log evaluation metrics
            if wandb_run:
                wandb_run.log({"eval/toxicity": avg_toxicity, "eval/epoch": epoch+1})
    
    # Save final model
    final_path = os.path.join(output_dir, "final-model")
    print(f"Saving final model to {final_path}")
    
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(final_path)
        
        # Save final reward stats
        reward_df = pd.DataFrame(reward_stats)
        reward_df.to_csv(os.path.join(output_dir, "final_reward_stats.csv"), index=False)
        
        # Push to Hugging Face Hub if enabled
        if cfg.output.push_to_hub:
            repo_name = cfg.output.repository_name or f"{cfg.model.name.split('/')[-1]}-detox"

            tokenizer.save_pretrained(final_path)
            with open(os.path.join(final_path, "rlhf_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
                        
            try:
                api = HfApi()
                owner = cfg.output.hf_owner
                repo_id = f"{owner}/{repo_name}"
                print(f"Pushing final model to Hugging Face Hub: {repo_id}")

                api.create_repo(repo_id=repo_id, private=getattr(cfg.output, "private", False),
                                exist_ok=True, repo_type="model")
                api.upload_folder(folder_path=final_path, repo_id=repo_id,
                                  commit_message="Final model after RLHF training")
                print(f"Successfully pushed model to {repo_id}")

            except Exception as e:
                print(f"Error pushing to Hugging Face Hub: {str(e)}")
    
    # Final evaluation
    final_toxicity, _ = evaluate_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="final"
    )
    
    print(f"Final evaluation: Average toxicity = {final_toxicity:.4f}")
    
    # Calculate total training time
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Training complete! Models and results saved to: {output_dir}")
    
    return final_toxicity


def safe_log_stats(ppo_trainer, stats, batch, rewards):
    """Safely log stats, handling NaN values."""
    clean_stats = {}
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            if np.isnan(v) or np.isinf(v):
                print(f"Warning: {k} has invalid value {v}, replacing with 0")
                clean_stats[k] = 0.0
            else:
                clean_stats[k] = v
        else:
            clean_stats[k] = v
    
    # Handle histograms
    for key in ['ppo/advantages', 'ppo/ratio', 'ppo/policy_loss', 'ppo/value_loss']:
        if key in clean_stats and isinstance(clean_stats[key], list):
            clean_stats[key] = [x for x in clean_stats[key] if isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)]
            if not clean_stats[key]:
                clean_stats[key] = [0.0]
    
    try:
        ppo_trainer.log_stats(clean_stats, batch, rewards)
    except Exception as e:
        print(f"Error in logging stats: {e}")
        try:
            minimal_stats = {
                'rewards/mean': clean_stats.get('rewards/mean', 0.0),
                'current_epoch': clean_stats.get('current_epoch', 0)
            }
            ppo_trainer.accelerator.log(minimal_stats)
        except Exception as e2:
            print(f"Even minimal logging failed: {e2}")


def safe_ppo_step(ppo_trainer, query_tensors, response_tensors, rewards):
    """Safely perform PPO step with error handling."""
    try:
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        return stats
    except RuntimeError as e:
        if "CUDA error" in str(e) or "device-side assert triggered" in str(e):
            print(f"CUDA error during PPO step: {e}")
            print("Returning empty stats dictionary")
            return {"error": str(e)}
        else:
            raise

