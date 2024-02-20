import torch
from torchvision import transforms

from dataclasses import dataclass
from trainer import Trainer
from src.config import InitInjectConfig
from src.utils import get_dataset_dataloader
from src.models import BahdanauCaptioner, LuongCaptioner, ParInjectCaptioner, InitInjectCaptioner, TransformerCaptioner


config = InitInjectConfig()
dataset, loader = get_dataset_dataloader(config.csv_file,
                                     config.transform,
                                     config.batch_size,
                                     config.max_length,
                                     config.freq_threshold)

model = InitInjectCaptioner(
    vocab=dataset.vocab,
    vocab_size=len(dataset.vocab),

    embed_dim=config.embedding_dim,
    encoder_dim=config.encoder_dim,
    decoder_dim=config.decoder_dim,
    num_layers=config.num_layers
)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


trainer = Trainer(config, dataset, loader, 
                  model, optimizer, criterion)
trainer.train(resume=False)