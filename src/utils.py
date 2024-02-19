import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .models import BahdanauCaptioner, LuongCaptioner, ParInjectCaptioner, InitInjectCaptioner, TransformerCaptioner
from .dataset import ImageCaptioningDataset

def load_model(path):
    name = path.split('/')[-1][:-4]
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    match name:
        case 'bahdanau':
            model = BahdanauCaptioner(
                vocab_size=checkpoint['vocab_size'],
                embed_dim=checkpoint['embed_dim'],
                attention_dim=checkpoint['attention_dim'],
                encoder_dim=checkpoint['encoder_dim'],
                decoder_dim=checkpoint['decoder_dim'],
                vocab=checkpoint['vocab']
            )
        case 'luong':
            model = LuongCaptioner(
                vocab_size=checkpoint['vocab_size'],
                embed_dim=checkpoint['embed_dim'],
                attention_dim=checkpoint['attention_dim'],
                encoder_dim=checkpoint['encoder_dim'],
                decoder_dim=checkpoint['decoder_dim'],
                vocab=checkpoint['vocab']
            )
        case 'par_inject':
            model = ParInjectCaptioner(
                vocab_size=checkpoint['vocab_size'],
                vocab=checkpoint['vocab'],
                embed_dim=checkpoint['embed_dim'],
                encoder_dim=checkpoint['encoder_dim'],
                decoder_dim=checkpoint['decoder_dim'],
                num_layers=checkpoint['num_layers'],
            )
        case 'init_inject':
            model = InitInjectCaptioner(
                vocab_size=checkpoint['vocab_size'],
                vocab=checkpoint['vocab'],
                embed_dim=checkpoint['embed_dim'],
                encoder_dim=checkpoint['encoder_dim'],
                decoder_dim=checkpoint['decoder_dim'],
                num_layers=checkpoint['num_layers'],
            )
        case 'transformer':
            model = TransformerCaptioner(
                n_tokens=checkpoint['n_tokens'],
                d_model=checkpoint['d_model'],
                n_heads=checkpoint['n_heads'],
                dim_forward=checkpoint['dim_forward'],
                n_layers=checkpoint['n_layers'],
                encoder_dim=checkpoint['encoder_dim'],
                vocab=checkpoint['vocab'],
            )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_dataset_dataloader(path, transform, batch_size):
    dataset = ImageCaptioningDataset(
                        csv_file=path,
                        transform=transform)

    loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2)

    return dataset, loader



def plot_result(image, caption):
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption)
    plt.show()

def plot_attention(image, result, attention_plot):
    result = result.split(" ")
    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7,7)

        ax = fig.add_subplot(len_result // 2,len_result // 2, l+1)
        ax.set_title(result[l])
        ax.set_axis_off()
        img = ax.imshow(image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())


    plt.tight_layout()
    plt.show()