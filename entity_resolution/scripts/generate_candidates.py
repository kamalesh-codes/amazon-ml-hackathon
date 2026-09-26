#!/usr/bin/env python3
import logging
import hydra
from omegaconf import OmegaConf
from entity_resolution.config import PipelineConfig
from entity_resolution.candidate_generation import generate_candidates

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
 c=PipelineConfig.from_mapping(OmegaConf.to_container(cfg, resolve=True)); logging.basicConfig(level=c.log_level,format="%(asctime)s %(levelname)s %(message)s"); generate_candidates(c)

if __name__=="__main__": main()
