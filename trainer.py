import torch
from tqdm import tqdm

from src.utils import get_dataset_dataloader
from src.models import BahdanauCaptioner, LuongCaptioner, ParInjectCaptioner, InitInjectCaptioner, TransformerCaptioner
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    def __init__(self):

        self.batch_size = 32
        self.num_epochs = 50

        self.embed_dim = 300
        self.encoder_dim = 512
        self.decoder_dim = 512
        self.num_layers = 4


        self.train_dataset, self.train_loader = get_dataset_dataloader("dataset/train.csv", self.batch_size)
        self.test_dataset, self.test_loader = get_dataset_dataloader("dataset/test.csv", self.batch_size)

        self.vocab_size = len(self.train_dataset.vocab)
        self.vocab = self.train_dataset.vocab

    def train(self, resume=False):
        # Init model, optimizer, criterion
        model = InitInjectCaptioner(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            vocab=self.train_dataset.vocab,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            num_layers=self.num_layers
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.train_dataset.vocab.word2index["<PAD>"])

        # Starting epoch
        start_epoch = 0

        if resume:
            # Load model and optimizer state
            model_state, optimizer_state, prev_epoch, prev_loss = self.load_model()
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

            # Starting epoch
            start_epoch = prev_epoch


        for epoch in range(start_epoch + 1, self.num_epochs + 1):

            model.train()
            train_epoch_loss = []
            train_pbar = tqdm(enumerate(iter(self.train_loader)), position=0, leave=True)

            for idx, (image, captions) in train_pbar:
                image, captions = image.to(device), captions.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Feed forward
                outputs = model(image, captions)

                # Calculate the loss
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, self.vocab_size), targets.reshape(-1))
                train_epoch_loss.append(loss.item())

                # Backward and update params
                loss.backward()
                optimizer.step()

                # Show progess bar
                train_pbar.set_postfix_str(f"{epoch}/{self.num_epochs} - Training loss: {sum(train_epoch_loss) / len(train_epoch_loss):0.4f}")

            # Test and save model
            self.test(model, optimizer, epoch, criterion)


    def test(self, model, optimizer, epoch, criterion):
        model.eval()
        test_pbar = tqdm(enumerate(iter(self.test_loader)), position=0, leave=True)
        test_epoch_loss = []
        for idx, (image, captions) in test_pbar:
            image, captions = image.to(device), captions.to(device)

            # Feed forward
            outputs = model(image, captions)

            # Calculate the loss
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, self.vocab_size), targets.reshape(-1))
            test_epoch_loss.append(loss.item())

            # Show progess bar
            test_pbar.set_postfix_str(f"{epoch}/{self.num_epochs} - Testing loss: {sum(test_epoch_loss) / len(test_epoch_loss):0.4f}")
        # Save model
        avg_test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
        self.save_model(model, optimizer, epoch, avg_test_epoch_loss)

    def save_model(self, model, optimizer, epoch, loss):
        model_state = {
            'epoch': epoch,
            'loss': loss,
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'embed_dim': self.embed_dim,
            'encoder_dim': self.encoder_dim,
            'decoder_dim': self.decoder_dim,
            'num_layers': self.num_layers,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(model_state, f'runs/models/init_inject.pth')

    def load_model(self):
        checkpoint = torch.load("runs/models/init_inject.pth")

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model_state = checkpoint['model_state_dict']
        optimizer_state = checkpoint['optimizer_state_dict']

        return model_state, optimizer_state, epoch, loss