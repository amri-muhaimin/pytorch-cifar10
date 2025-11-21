# CIFAR-10 Classification with PyTorch (GPU)

Proyek ini berisi contoh training dan testing model CNN sederhana untuk dataset CIFAR-10
menggunakan PyTorch dan GPU (CUDA).

## Struktur Proyek

- `train_cifar10.py`  
  Script utama untuk melakukan training model pada dataset CIFAR-10.

- `test_cifar10.py`  
  Script untuk melakukan evaluasi model yang sudah dilatih menggunakan data test CIFAR-10.

- `data/`  
  Folder dataset CIFAR-10. Akan dibuat dan diisi **secara otomatis** oleh `torchvision`
  saat `train_cifar10.py` pertama kali dijalankan.  
  Folder ini **tidak di-push ke GitHub** (sudah masuk `.gitignore`).

- `checkpoints/`  
  Folder untuk menyimpan model `.pt` terbaik hasil training.  
  Folder ini juga **tidak di-push ke GitHub**.

## Persiapan Environment

Contoh menggunakan conda (environment `ds-gpu`):

```bash
conda activate ds-gpu
pip install torch torchvision torchaudio
