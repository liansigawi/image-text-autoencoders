<h1 align="center">Autoencoders for Image Reconstruction and Text Denoising</h1><p align="center"><strong>Reconstruct handwritten digits, restore corrupted text, and reuse a text encoder for sentiment classification.</strong></p><p align="center">
  <img src="https://img.shields.io/badge/Python-1D4ED8?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-B93822?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/MNIST-6D28D9?style=for-the-badge" alt="MNIST">
  <img src="https://img.shields.io/badge/IMDb-0F766E?style=for-the-badge" alt="IMDb">
</p><p align="center">
  <a href="#image-reconstruction">Image reconstruction</a> ·
  <a href="#text-denoising">Text denoising</a> ·
  <a href="#sentiment-classification">Sentiment classification</a> ·
  <a href="#run-locally">Run locally</a>
</p>Image reconstruction
I trained symmetric autoencoders on MNIST with a loss that combines image reconstruction error and a layer-wise penalty. The penalty encourages each decoder layer to recover the activations that went into its matching encoder layer. I compared three fully connected architectures and also trained a convolutional autoencoder.

Fully connected encoder widths	Bottleneck size	Combined test loss ↓
[512, 128]	128	0.0098
[256, 64, 16]	16	0.0326
[128, 64, 32, 8]	8	0.0489


The shallowest, widest model had the lowest combined test loss after five epochs. In the examples above, the model with the 8-dimensional bottleneck retained the general shape of the digits but blurred fine strokes. Depth and bottleneck width varied together in this experiment, so their effects cannot be separated from these runs.
Text denoising
I built a GRU sequence-to-sequence autoencoder that learns to reconstruct IMDb review excerpts after some tokens are masked or dropped. I first checked that the model could learn a fixed batch of eight examples. Then I trained on 10,000 excerpts for 50 epochs and inspected reconstructions from the test split without teacher forcing. The model recovered some excerpts exactly and repeated or substituted words in others. This part of the notebook reports qualitative examples.
Sentiment classification
I reused the denoising encoder for binary sentiment classification and compared keeping its weights frozen with fine-tuning them. Both strategies started with the same pretrained encoder and were evaluated on a balanced subset of 2,000 IMDb test examples.
Training strategy	Test accuracy
Frozen encoder	56.30%
Fine-tuned encoder	63.85%


Fine-tuning improved test accuracy by 7.55 percentage points in this run. The comparison tests how to reuse the pretrained encoder. A classifier trained with a randomly initialized encoder would be needed to measure the benefit of denoising pretraining itself.
Run locally
Open the Jupyter notebook in this repository to see the saved outputs. To rerun it, install the dependencies and launch JupyterLab from the repository folder.
python -m pip install torch torchvision datasets matplotlib numpy jupyterlab
jupyter lab
Run the notebook cells in order. The first run downloads MNIST and IMDb. The saved training run used CUDA, which is recommended for the longer GRU experiment.
Developed as part of the Deep Learning course at the Open University of Israel
