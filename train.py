import torch
from torchvision import transforms

from dataclasses import dataclass
from trainer import Trainer
from src.config import InitInjectConfig
from src.utils import get_dataset_dataloader
from src.models import BahdanauCaptioner, LuongCaptioner, ParInjectCaptioner, InitInjectCaptioner, TransformerCaptioner


dataset, loader = get_dataset_dataloader(InitInjectConfig.csv_file,
                                     InitInjectConfig.transform,
                                     InitInjectConfig.batch_size,
                                     InitInjectConfig.max_length,
                                     InitInjectConfig.freq_threshold)

model = InitInjectCaptioner(
    vocab=dataset.vocab,
    vocab_size=len(dataset.vocab),

    embed_dim=InitInjectConfig.embedding_dim,
    encoder_dim=InitInjectConfig.encoder_dim,
    decoder_dim=InitInjectConfig.decoder_dim,
    num_layers=InitInjectConfig.num_layers
)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


trainer = Trainer(InitInjectConfig, dataset, loader, 
                  model, optimizer, criterion)
trainer.train(resume=False)