#!/usr/bin/env python3
"""
RLHF Training Script
Trains language models using Reinforcement Learning from Human Feedback (RLHF)
for text detoxification.
"""

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from irl_pipeline.rlhf.train import train_rlhf


@hydra.main(config_path="../configs", config_name="full_pipeline.yaml", version_base=None)
def main(cfg: DictConfig):
    
    # 1) take RLHF node from full pipeline
    rlhf_cfg = cfg.rlhf if "rlhf" in cfg else cfg

    # Safely copy root IRL block into rlhf_cfg even if struct-locked
    if "irl" in cfg and "irl" not in rlhf_cfg:
        with open_dict(rlhf_cfg):
            rlhf_cfg["irl"] = cfg.irl
    
    print("="*60)
    print("Starting RLHF Training")
    print("="*60)
    
    # Train the model
    final_toxicity = train_rlhf(rlhf_cfg)
    
    print("="*60)
    print(f"Training Complete! Final toxicity: {final_toxicity:.4f}")
    print("="*60)
    
    return final_toxicity


if __name__ == "__main__":
    main()




# # Basic RLHF training
# ./run_pipeline.sh rlhf

# # With specific model
# ./run_pipeline.sh rlhf rlhf_config.model=llama_3_2_1b

# # With IRL reward
# ./run_pipeline.sh rlhf \
#   rlhf_config.irl.posterior_dir=outputs/re_irl/.../round_1 \
#   rlhf_config.irl.global_norm_dir=outputs/re_irl/...