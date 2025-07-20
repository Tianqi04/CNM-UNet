# CNM-UNet: Continuous Ordinary Differential Equations for Medical Image Segmentation

**0. Main Environments**
- python 3.8
- [pytorch 1.8.0](https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-win_amd64.whl)
- [torchvision 0.9.0](https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl)

**1. Prepare the dataset.**

- The ISIC18, Polyp, and OD/OC datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1fxFPTyHtLiMPvbC8RX19lYNk-xPH8gW-). 
- After downloading, please place the datasets into the following directories:

  - './ISIC18/data/ISIC18/'
  - './Polyp/data/Polyp/'
  - './ODOC/data/ODOC/'

**2. Configuration.**
- All experiment parameters can be modified within the file config_setting.py.

**3. Model Training and Source Set Performance Testing (Polyp Dataset Example).**

```bash
cd Polyp
python train.py
```

**4. Generalization Capability Testing (Polyp Dataset Example).**
- To test the generalization capability, place your pre-trained model into the ./Polyp/results/ . 
- Remember to modify the pre-trained model path and test sets name in config_setting.py.

```bash
cd Polyp
python test.py
```

**5. Obtains theResults (Polyp Dataset Example).**
- After training or testing, you could obtain the results in './Polyp/results/'.

**6. Acknowledgement.**
- Parts of the code are based on the Pytorch implementations of [Lightweight nmODE](https://github.com/nayutayuki/Lightweight-nmODE-Decoders-For-U-like-networks), [NODE](https://github.com/rtqichen/torchdiffeq) and [VPTTA](https://github.com/Chen-Ziyang/VPTTA).
