import torch
from tqdm import tqdm

from src.utils import get_dataset_dataloader
from src.models import BahdanauCaptioner, LuongCaptioner, ParInjectCaptioner, InitInjectCaptioner, TransformerCaptioner
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    def __init__(self, config, dataset, loader, model, optimizer, criterion):
        self.config = config
        self.dataset = dataset
        self.loader = loader
        self.vocab_size = len(self.dataset.vocab)
        self.vocab = self.dataset.vocab

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion

    def run_epoch(self, epoch):
        self.model.train()
        epoch_loss = []
        pbar = tqdm(enumerate(iter(self.loader)), position=0, leave=True)

        for idx, (image, captions, targets) in pbar:
            image, captions, targets = image.to(device), captions.to(device), targets.to(device)

            self.optimizer.zero_grad() # Zero the gradients
            outputs = self.model(image, captions) # Forward

            # Calculate loss
            loss = self.criterion(outputs.view(-1, self.vocab_size), targets.reshape(-1))
            epoch_loss.append(loss.item())

            # Backward and update params
            loss.backward()
            self.optimizer.step()

            # Show progess bar
            pbar.set_postfix_str(f"Epoch: {epoch}/{self.config.num_epochs} - Loss: {sum(epoch_loss) / len(epoch_loss):0.4f}")

    def train(self, resume=False):
        start_epoch = 0
        if resume:
            # Load model and optimizer state
            model_state, optimizer_state, prev_epoch, prev_loss = self.load_model()
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            start_epoch = prev_epoch # Starting epoch

        for epoch in range(start_epoch + 1, self.config.num_epochs + 1):
            epoch_loss = self.run_epoch(epoch)
            
            # Save model
            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.save_model(self.model, self.optimizer, epoch, avg_epoch_loss)


    def save_model(self, model, optimizer, epoch, loss):
        model_state = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),

            'vocab': self.vocab,
            'vocab_size': self.vocab_size,

            'embed_dim': self.config.embedding_dim,
            'encoder_dim': self.config.encoder_dim,
            'decoder_dim': self.config.decoder_dim,
            'num_layers': self.config.num_layers,
        }

        torch.save(model_state, f'models/init_inject_4lstm/model_best.pth')

    def load_model(self):
        checkpoint = torch.load("models/init_inject_4lstm/model_best.pth")

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model_state = checkpoint['model_state_dict']
        optimizer_state = checkpoint['optimizer_state_dict']

        return model_state, optimizer_state, epoch, loss