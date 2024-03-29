# Image Captioning

Image Captioning is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing

# Architectures

<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/98479846-a9e6-4811-8c14-67c09469cd9f.png" alt="drawing" width="75%" height="75%"/>
</p>

* init-inject: The image vector is treated as an initial hidden state vector
for the RNN. After initialising the RNN, the vectors in the caption prefix
are then fed to the RNN as usual.

* pre-inject: The image vector is used as the first ‘word’ in the caption
prefix. This makes the image vector the first input that the RNN will see.

* par-inject: The image vector is concatenated to every word vector in the
caption prefix in order to make the RNN take a mixed word-image vector.
Every word would have the exact same image vector concatenated to it.

* merge: The image vector and caption prefix vector are concatenated into
a single vector before being fed to the output layer.

# Dataset
* A new benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations
* Each image in the dataset will have 5 captions. Example:

<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/ca86706d-6009-41e7-9cdb-13c4f6c73f75.png" alt="drawing" width="50%" height="50%"/>
</p>
<br/>
<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/fbbd005a-fb62-43c8-9940-d15b2074d63f.png" alt="drawing" width="50%" height="50%"/>
</p>



# Bahdanau Attention
<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/b30f36fb-52c6-40b7-9d91-6a2ee1fdb555" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Bahdanau Attention</a>
</p>
<br/>

# Luong Attention

<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/a094347e-6033-4aff-9f95-5de21df91289" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Luong Attention</a>
</p>

# Transformer
<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/c81c1875-b1c4-4f7b-b6a5-7bf3aaa5bc0d" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Transformer Architecture</a>
</p>

In image captioning, we will use only the decoder part of the transformer for caption generation. We won't have a encoder so we replace it with a CNN model which will extract features from image and feed it into decoder as memory
# ViT
<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/149d363e-ab27-4955-9d9b-7454d8e6147f" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Vison Transformer (ViT)</a>
</p>


<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/1b83ce28-225c-4ce4-8d8a-704971f579a2" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">ViT-GPT2</a>
</p>


  
# BLEU SCORE

BLEU (Bilingual Evaluation Understudy) is a measurement of the difference between an automatic translation and human-created reference translations of the same source sentence.

The BLEU algorithm compares consecutive phrases of the automatic translation with the consecutive phrases it finds in the reference translation, and counts the number of matches, in a weighted fashion. These matches are position independent. A higher match degree indicates a higher degree of similarity with the reference translation, and higher score. Intelligibility and grammatical correctness aren't taken into account.

<div align="center">
  
| Model                |   bleu-4         |  bleu-3      | bleu-2      |  bleu-1      | 
|----------------------|:----------------:|:------------:|:-----------:|:------------:|
| Bahdanau Attention   |  0.5529          | 0.6401       | 0.7334      | 0.8418       | 
| Luong Atttention     |  0.5889          | 0.6605       | 0.7410      | 0.8433       |    
| Transformer          |  0.3735          | 0.4562       | 0.5334      | 0.6046       | 
| Transformer-v2       |  **0.6924**      | **0.7738**   | **0.8499**  | **0.9169**   | 
| Par-Inject           |  0.34            |              |             |              |  
| Par-Inject-4-LSTM    |  0.3942          | 0.4804       | 0.5889      | 0.7408       |    
| Init-Inject          |  0.27            |              |             |              |
| Init-Inject-4-LSTM   |  0.5542          | 0.6031       | 0.6679      | 0.7740       |
| ViT-GPT2             |  0.5202          | 0.5960       | 0.6759      | 0.7622       |
</div>


# TODO
* Quantization

# Sample
<table>
    <thead>
        <tr>
            <th>Image</th>
            <th>Model</th>
            <th>Generated caption</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=7 width=50%>
              <p align="center">
                <img src="https://github.com/ptatien0307/image_captioning/assets/79583501/bcec898f-4d43-4711-861b-fc686b514ba8" alt="drawing" width="100%" height="100%"/>
              </p>
            </td>
            <td>Bahdanau</td>
            <td>a girl in a pink swimsuit is laying in the water</td>
        </tr>
        <tr>
            <td>Luong</td>
            <td>a girl is stretched out in shallow water</td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>a man in a red shirt is riding a skateboard in a pool</td>
        </tr>
        <tr>
            <td>Transformer-v2</td>
            <td>a girl is laying on her back in shallow water</td>
        </tr>
        <tr>
            <td>Par-Inject-4LSTM</td>
            <td>a girl in a bikini lying on her back in shallow water</td>
        </tr>
        <tr>
            <td>Init-Inject</td>
            <td>a girl is stretched out in water , and her hair is taking a breath .</td>
        </tr>
       <tr>
            <td>ViT-GPT2</td>
            <td>A girl in a bikini lays on the beach .</td>
        </tr>
          <tr>
            <td rowspan=7 width=50%>
              <p align="center">
                <img src="https://github.com/ptatien0307/image_captioning/assets/79583501/938b4441-1665-473e-ad13-65ddcfb14241" alt="drawing" width="100%" height="100%"/>
              </p>
            </td>
            <td>Bahdanau</td>
            <td>a man and a girl horse and a horse is standing next to a fire</td>
        </tr>
        <tr>
            <td>Luong</td>
            <td>a man and a woman are standing on a fire with a fire in the background</td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td>a man and a female stand around a fire</td>
        </tr>
        <tr>
            <td>Transformer-v2</td>
            <td>a girl and her horse stand by a fire</td>
        </tr>
        <tr>
            <td>Par-Inject-4LSTM</td>
            <td>a dog is running in a field of a field</td>
        </tr>
        <tr>
            <td>Init-Inject</td>
            <td>a horse and rider , surrounded by several birds standing near a horse</td>
        </tr>
        <tr>
            <td>ViT-GPT2</td>
            <td>A woman and a young boy are sitting in a fire .</td>
        </tr>
    </tbody>
</table>


# Web
<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/b44a88bb-c7f1-4631-a850-28d2145dffc7" alt="drawing" width="75%" height="75%"/>
<br/>
</p>


