import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast

from .models import BahdanauCaptioner, LuongCaptioner, ParInjectCaptioner, InitInjectCaptioner, TransformerCaptioner
from .dataset import ICDatasetTransformer, ICDataset

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

def load_vit_gpt(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    tokenizer = GPT2TokenizerFast.from_pretrained(path)
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    return model, tokenizer, image_processor


def get_dataset_dataloader(file, transform, batch_size, max_length, freq_threshold):
    dataset = ICDataset(
                        csv_file=file,
                        transform=transform,
                        max_length=max_length,
                        freq_threshold=freq_threshold)

    dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2)

    return dataset, dataloader



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