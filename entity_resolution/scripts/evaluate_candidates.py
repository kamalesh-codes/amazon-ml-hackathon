#!/usr/bin/env python3
import json
import hydra
from omegaconf import OmegaConf
from entity_resolution.config import PipelineConfig
from entity_resolution.candidate_generation import evaluate

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
 c=PipelineConfig.from_mapping(OmegaConf.to_container(cfg, resolve=True))
 print(json.dumps(evaluate(c.candidates_path,c.ground_truth_path,c.evaluation_path),indent=2))

if __name__=="__main__": main()
