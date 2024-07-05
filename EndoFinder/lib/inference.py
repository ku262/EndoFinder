
import logging
import os
import torch
import pytorch_lightning as pl
import numpy as np
from classy_vision.generic.distributed_util import get_rank, get_world_size, barrier
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.plugins import DDPSpawnPlugin
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

from torchvision.models import resnet50, vgg19, densenet121
from timm import create_model as creat


from EndoFinder.tools.img2jsonl import LoadModelSetting
from EndoFinder.lib.util import call_using_args, parse_bool


logger = logging.getLogger("inference.py")
logger.setLevel(logging.INFO)

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

def dedup_outputs(outputs, key="instance_id"):
    """Deduplicate dataset on instance_id."""
    idx = np.unique(outputs[key].numpy(), return_index=True)[1]
    outputs = {k: v.numpy()[idx] for (k, v) in outputs.items()}
    assert np.unique(outputs[key]).size == outputs["instance_id"].size
    return outputs

class InferenceModel(pl.LightningModule):
    """Wraps a model for inference."""

    def __init__(self, model, metadata_keys):
        super().__init__()
        self.model = model
        self.metadata_keys = metadata_keys

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        input = batch["input"]
        batch = {k: v for (k, v) in batch.items() if k in self.metadata_keys}
        batch["embeddings"] = self(input)

        # Workaround for a CUDA synchronization bug in PyTorch Lightning.
        # Fixed upstream:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11287
        batch = {k: v.cpu() for (k, v) in batch.items()}

        return batch


class Inference:
    @classmethod
    def add_parser_args(cls, parser):
        parser.add_argument("--model_state", required=True)
        parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
        parser.add_argument("--output_path", required=True)
        parser.add_argument("--gpus", default=4)
        parser.add_argument("--accelerator", default="auto")
        parser.add_argument("--nodes", default=1, type=int)
        parser.add_argument("--workers", default=10, type=int)
        parser.add_argument(
            "--size", default=224, type=int, help="Image size for inference"
        )
        parser.add_argument("--preserve_aspect_ratio", default=False, type=parse_bool)
        # These options are only used if --model_state is provided.


    @classmethod
    def inference(cls, args, dataset, base_name="predictions"):
        # if args.model_state:
        logger.info("Loading model")
        if args.model == 'vit_large_patch16':
            model = LoadModelSetting.get_model(LoadModelSetting[args.model], args.model_state)
            # model = torch.jit.load(args.model_state)
        pl_model = InferenceModel(model, ["image_num", "split", "instance_id"])

        logger.info("Creating dataloader")
        dataloader = DataLoader(
            dataset,
            batch_size=1 if args.preserve_aspect_ratio else 256,
            num_workers=args.workers,
            persistent_workers=(
                args.workers > 0
            ),  # unnecessary here, but silences warning
        )
        writer = InferenceWriter(args.output_path, base_name)
        trainer = pl.Trainer(
            devices=args.gpus,
            num_nodes=args.nodes,
            accelerator=args.accelerator,
            default_root_dir=args.output_path,
            strategy=DDPSpawnPlugin(find_unused_parameters=False),
            callbacks=[writer],
            log_every_n_steps=1,
        )
        logger.info("Starting inference")
        trainer.predict(pl_model, dataloaders=dataloader)
        logger.info("Loading features")
        outputs = writer.read()

        logger.info("Deduplication")
        outputs = dedup_outputs(outputs)
        return outputs


def coalesce_outputs(outputs):
    keys = outputs[0].keys()
    return {k: torch.cat([out[k] for out in outputs]) for k in keys}


class InferenceWriter(BasePredictionWriter):
    def __init__(self, output_path: str, filename: str):
        super().__init__("epoch")
        self.output_path = output_path
        self.filename = filename
        self.output_file = os.path.join(self.output_path, f"{filename}.pt")

    def _rank_fn(self, i):
        return os.path.join(self.output_path, f"{self.filename}_rank_{i}.pt")

    def write_on_epoch_end(self, trainer, module, predictions, batch_indices):
        rank = get_rank()
        assert len(predictions) == 1
        predictions = predictions[0]
        outputs = coalesce_outputs(predictions)
        logger.info(
            "Writing %d outputs for worker %d", outputs["embeddings"].size(0), rank
        )
        torch.save(outputs, self._rank_fn(rank))
        del outputs
        logger.info("Rank %d done. Waiting for peers.", rank)
        barrier()
        if rank == 0:
            logger.info("Combining prediction outputs.")
            worker_output_fns = [self._rank_fn(i) for i in range(get_world_size())]
            worker_outputs = [torch.load(fn) for fn in worker_output_fns]
            outputs = coalesce_outputs(worker_outputs)
            del worker_outputs
            torch.save(outputs, self.output_file)
            logger.info("Save completed.")
            for fn in worker_output_fns:
                os.remove(fn)

    def read(self):
        return torch.load(self.output_file)
