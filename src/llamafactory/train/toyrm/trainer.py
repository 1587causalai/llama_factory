import torch
from typing import Dict, List, Optional, Tuple, Union

from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from ...extras import logging

logger = logging.get_logger(__name__)

class ToyRewardModelTrainer(Trainer):
    def __init__(self, finetuning_args=None, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )[0] # [batch_size, 1]
        
        # Modify the original loss by dividing by 1000 and adding 1
        loss = -torch.log(torch.sigmoid(rewards[::2] - rewards[1::2])) / 1000.0 + 1.0
        loss = loss.mean()

        if return_outputs:
            return loss, {"rewards": rewards}
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
        loss = loss.mean().detach()
        rewards = outputs["rewards"]

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, rewards, inputs["input_ids"]) 