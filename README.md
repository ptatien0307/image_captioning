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


# Models
<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/b30f36fb-52c6-40b7-9d91-6a2ee1fdb555" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Bahdanau Attention</a>
</p>
<br/>


<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/a094347e-6033-4aff-9f95-5de21df91289" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Luong Attention</a>
</p>



<p align="center">
<img src="https://github.com/ptatien0307/image_captioning/assets/79583501/f7848525-d76a-4997-a980-d3a5e3aab045" alt="drawing" width="50%" height="50%"/>
<br/>
<a style="text-align: center">Par-Inject</a>
</p>





# TODO
* Luong Attention
* Bahdanau Atttention
* Teaching forcing

# BLEU SCORE
Bahdanau Attention: 0.41
