
<!-- ## <div align="center"><b>Thermal2Sense</b></div> -->

<div align="center">
  
## Thermal2Sense: A new method for reconstructing Thermal images into RGB images using semantic information from large-scale diffusion models
[📄[Paper]()] &emsp; [🚀[Gradio Demo](http://lst-showroom.natapp1.cc)] &emsp; <br>

</div>


---
## 🔥 **Examples**

<p align="center">
  <img src="https://github.com/no996no007/Thermal2Sense/assets/135965025/75e38c30-36cf-4509-bf70-84f3116086a5" height=250>
</p>


## 🏷️ Introduce

In the fields of security monitoring and autonomous driving, low-light conditions pose significant challenges to the performance of visual sensors. To mitigate this issue, this paper presents a method for converting infrared to visible light images called Thermal2Sense. The approach is based on the technical routes of diffusion models and Generative Adversarial Networks (GANs) and utilizes a two-stage inference process to optimize the quality of image reconstruction. In the first stage, Thermal2Sense employs a dedicated Controlnet module to reconstruct images with random noise input, ensuring structural consistency with the infrared image. The second stage additionally introduces an IP-Adapter module and Canny Controlnet, leveraging semantic knowledge from pre-trained models for semantic reconstruction at the detail level while maintaining overall structural and color control.
<p align="center">
  <img src="https://github.com/no996no007/Thermal2Sense/assets/135965025/b0abef6b-c629-4a9e-ba74-6a95df7d27f2" height=250>
</p>



## 🔧 Requirements

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
- cuda==11.8

```bash
conda create --name Thermal2Sense python=3.8.10
conda activate Thermal2Sense
pip install -U pip

# Install requirements
pip install -r requirements.txt
```

### Infer
```setup
python infer.py
```

### Gradio Demo
```setup
python app.py
```


## ⏬ Model weights

The parameters of the IPAdapter and Controlnet we trained for the T2V task can be downloaded from the following link：[Baidu Netdisk](https://pan.baidu.com/s/1toBfPoF6b2-z_PYO8PpaaQ?pwd=ip7r).


## Acknowledgement
* Inspired from many excellent demos and repos, including [IPAdapter](https://github.com/tencent-ailab/IP-Adapter), [ControlNet](https://github.com/lllyasviel/ControlNet). Thanks for their great work!
* Thanks to the open source contributions of the following work: [LLaVA](https://github.com/haotian-liu/LLaVA). 




