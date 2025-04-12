# Face Generator using DCGAN (CelebA Dataset)
![gif](assets\DCGAN.gif)

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate realistic human faces. It uses the CelebA dataset containing 202,599 celebrity images and is modularized for flexibility—so you can train on your own dataset too!

## 🧠 Highlights

* Built with PyTorch and trained on the CelebA dataset
* Modular code: Easy to swap datasets, tune hyperparameters, or modify architectures
* Supports training with your own dataset
* Saves generated samples during training so you can track progress visually

## 📁 Project Structure

```
.
├── config.py                # Global constants (e.g., image size, paths)
├── dataset.py               # Custom PyTorch Dataset class
├── models.py                # Generator and Discriminator
├── train.py                 # Main training script
├── utils.py                 # Utilities for saving images, visualizing outputs
├── saved_images/            # Output folder for generated image grids
└── requirements.txt         # Dependencies
```

## 🖼️ Sample Output

Generated samples during training:

![image](saved_images\image_23500.png)

More samples get saved automatically in the `saved_images/` folder.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/manohar3000/DCGAN.git
cd DCGAN
```

### 2. Install Requirements

Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

### 3. Download the CelebA Dataset

* Download the **img_align_celeba** dataset from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* Extract it and place it in your desired directory.
* Update the `DATASET_PATH` in `config.py` to point to the dataset location.

### 4. Start Training

```bash
python train.py
```

## 📂 Using Your Own Dataset

You can train this GAN on **any dataset of images**.

1. Place your images (JPG/PNG) in a folder like `./your_dataset/`
2. Update `DATASET_PATH` in `config.py`:

```python
DATASET_PATH = "your_dataset/"
```

3. Make sure your images are in RGB format and have roughly similar dimensions (they will be resized to 64x64 anyway).

That's it! Run `python train.py` to begin training on your dataset.

## ⚙️ Configuration (config.py)

You can modify:

```python
BATCH_SIZE = 128
IMAGE_SIZE = 64
Z_DIM = 100  # Latent vector size
NUM_EPOCHS = 5
LEARNING_RATE = 0.0002
DATASET_PATH = "D:/img_align_celeba"
DEVICE = "cuda" or "cpu"
```

## 📈 Training Progress

![image](assets\loss_plot.png)

* Generator and Discriminator losses are logged to the console
* Images are saved every 500 iterations
* Modify the frequency of logging/saving in `train.py` if needed

## 🧠 Credits

* **DCGAN Architecture**: Based on Radford et al. (2016)
* **Dataset**: CelebFaces Attributes (CelebA)

## 🛠️ Troubleshooting

* **CUDA error**: If you don't have a GPU, training will use CPU (but it's slower).
* **Black/Noisy Images?** Let it train longer. You'll see improvement over time.
* **Memory issues?** Try reducing `BATCH_SIZE` in `config.py`.

## 👥 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, or submit PRs if you'd like to:
* Add new generator/discriminator variants
* Improve training strategies
* Support additional datasets

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more information.
