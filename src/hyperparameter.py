import logging
import os

import hydra
import omegaconf
import torch
from hyperparameter_tuning.hyperparameter_tuning import HyperparameterTuning
from utils.general_utils import setup_logging
from utils.seed_utils import fix_seed


@hydra.main(version_base=None, config_path="../conf", config_name="hyperparameter.yaml")
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    if cfg.environ.seed:
        fix_seed(seed=cfg.environ.seed)
        logger.info(f"Seed fixed at {cfg.environ.seed}.\n")

    if cfg.environ.device == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.environ.device}")
    logger.info(f"Device set to {device}.\n")

    tune_hyperparameters = HyperparameterTuning(cfg=cfg, logger=logger, device=device)
    tune_hyperparameters.perform_hyperparameter_tuning()


if __name__ == "__main__":
    main()
