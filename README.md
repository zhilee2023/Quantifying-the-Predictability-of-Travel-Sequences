
## ⚙Official Implementations for *Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder*

This repository provides the code associated with the paper *Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder* (VQ-VAE). The code primarily focuses on the first experiment of the paper, which investigates the generation of two-dimensional Gaussian-Markov sequences and the application of VQ-VAE to model the predictability of these sequences.

## 📦 Dependencies

The main dependencies for this project include:

- `numpy`: Numerical computing
- `scipy`: Scientific computation tools
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `torch`: PyTorch deep learning framework
- `torchvision`, `torchaudio`: Optional PyTorch components for image/audio support
- `triton`: Accelerated kernel backend (used by PyTorch)

All dependencies are listed in `requirements.txt`.

---

## 🔧 Installation

To install all required Python packages:

```python
pip install -r requirements.txt
```

## 🧪 Experiment 1: Gaussian-Markov Sequence Generation and Predictability Analysis

In this experiment, we generate synthetic 2D Gaussian-Markov sequences that emulate stylized travel patterns. These sequences serve as the input to our **Vector-Quantized Variational Autoencoder (VQ-VAE)** model, which quantifies behavioral predictability through the estimation of rate-distortion curves.

- `data_gen.py`: Generates 2D Gaussian-Markov sequences.
- `experiment1.py`: Applies VQ-VAE to the generated data and plots the rate-distortion curve.

### 📊 Dataset Generation

The synthetic dataset simulates a discrete-time Gaussian-Markov process with the following characteristics:

- **Transition Matrix**: Specifies the linear dependency between consecutive states.
- **Mean and Covariance**: Defines the parameters of the Gaussian distribution from which the noise is sampled at each time step.

To generate a new dataset:

```python
python data_gen.py
```


### 📈 Running the VQ-VAE and Plotting Rate–Distortion

Once the dataset is generated, run the following script to apply the VQ-VAE model and produce a rate-distortion curve:

```python
python experiment_1.py
```

## 📄 Reference
If you use this code or experiment in your research, please cite:
```bibtex
@misc{Li2025,
  author       = {Zhi Li and Zhibin Chen and Minghui Zhong},
  title        = {Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder},
  note         = {Preprint, under review},
  year         = {2025},
}
