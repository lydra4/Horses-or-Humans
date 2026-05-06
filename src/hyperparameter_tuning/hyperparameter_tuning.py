import copy
import logging
import os
from typing import Dict, Optional, Tuple

import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from optuna.samplers import GridSampler
from sklearn.model_selection import ParameterGrid
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

from training_pipelines.training_pipelines import TrainingPipeline


class HyperparameterTuning(TrainingPipeline):
    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[logging.Logger],
        device: torch.device,
    ):
        self.full_cfg: DictConfig = cfg
        self.cfg = copy.copy(cfg)
        super().__init__(cfg=self.cfg, logger=logger, device=device)

        self.logger = logger or logging.getLogger(__name__)
        self.device = device

        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.current_learning_rate: Optional[float] = None
        self.current_batch_size: Optional[int] = None
        self.current_optimizer_name: Optional[str] = None

    def _initialize_efficientnet(self):
        self.logger.info("Loading Efficientnet Model.\n")

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        number_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(
            in_features=number_features, out_features=self.full_cfg.out_features
        )

        self.model = self.model.to(self.device)
        self.logger.info(f"Efficientnet Model loaded to {self.device}.\n")

    def _set_trials(self, trial: optuna.Trial) -> Tuple[float, int, str]:
        learning_rate = trial.suggest_categorical(
            "learning_rate", self.full_cfg.learning_rate
        )
        batch_size = trial.suggest_categorical("batch_size", self.full_cfg.batch_size)
        optimizer_name = trial.suggest_categorical(
            "optimizer_name", self.full_cfg.optimizer_name
        )

        return learning_rate, batch_size, optimizer_name

    def _update_config(
        self,
        learning_rate: float,
        batch_size: int,
        optimizer_name: str,
    ):
        self.cfg = copy.copy(self.full_cfg)
        self.cfg["learning_rate"] = learning_rate
        self.cfg["batch_size"] = batch_size
        self.cfg["optimizer_name"] = optimizer_name

        self.current_learning_rate = learning_rate
        self.current_batch_size = batch_size
        self.current_optimizer_name = optimizer_name

    def _initialize_optimizer_and_criterion(
        self, params
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        self.criterion = nn.CrossEntropyLoss()

        if self.cfg.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(params=params, lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(params=params, lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(params=params, lr=self.cfg.learning_rate)
        else:
            raise ValueError(f"Unsupported Optimizer: {self.current_optimizer_name}")

        self.logger.info(f"Using {self.current_optimizer_name} optimizer.\n")

        return self.criterion, self.optimizer

    def _objective(self, trial: optuna.Trial) -> float:
        learning_rate, batch_size, optimizer_name = self._set_trials(trial=trial)

        self._update_config(
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
        )

        self.cfg.checkpoint_save_path = os.path.join(
            self.full_cfg.checkpoint_save_path,
            optimizer_name,
            f"batch_size_{str(batch_size)}",
            f"learning_rate_{str(learning_rate)}",
        )

        self._set_transforms()
        self._load_dataset_from_folder()
        self._initialize_efficientnet()
        if self.model is None:
            raise RuntimeError(
                "Model failed to initialize in _initialize_efficientnet."
            )
        self._initialize_optimizer_and_criterion(params=self.model.parameters())
        self._setup_mlflow()
        best_val_accuracy = self._train_model()

        trial.set_user_attr("learning_rate", learning_rate)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("optimizer_name", optimizer_name)
        trial.set_user_attr("val_accuracy", best_val_accuracy)

        return best_val_accuracy

    def _set_search_space(self) -> Dict[str, list]:
        search_space = {
            "learning_rate": self.full_cfg.learning_rate,
            "batch_size": self.full_cfg.batch_size,
            "optimizer_name": self.full_cfg.optimizer_name,
        }

        return search_space

    def perform_hyperparameter_tuning(self) -> None:
        search_space = self._set_search_space()
        total_trials = len(list(ParameterGrid(search_space)))
        self.logger.info(
            f"Starting hyperparameter tuning with {total_trials} combinations.\n"
        )

        db_path = self.full_cfg.optuna_storage_path
        storage = f"sqlite:///{db_path}"

        sampler = GridSampler(
            search_space=search_space, seed=(self.full_cfg.environ.seed or 42)
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        for _ in tqdm(range(total_trials), desc="Hyperparameter Tuning"):
            study.optimize(self._objective, n_trials=1, catch=(Exception,))
