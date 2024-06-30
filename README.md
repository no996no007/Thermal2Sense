
<!-- ## <div align="center"><b>Thermal2Sense</b></div> -->

<div align="center">
  
## Thermal2Sense: A new method for reconstructing Thermal images into RGB images using semantic information from large-scale diffusion models
[📄[Paper]()] &emsp; [🚀[Gradio Demo](http://lst-showroom.natapp1.cc)] &emsp; <br>

</div>


---
## 🔥 **Examples**

<p align="center">
  <img src="https://github.com/JackAILab/ConsistentID/assets/135965025/f949a03d-bed2-4839-a995-7b451d8c981b" height=450>
</p>


## 🏷️ Introduce
- [![Huggingface ConsistentID](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/JackAILab/ConsistentID)
- [**ConsistentID Model Card**](https://huggingface.co/JackAILab/ConsistentID)
  
In the fields of security monitoring and autonomous driving, low-light conditions pose significant challenges to the performance of visual sensors. To mitigate this issue, this paper presents a method for converting infrared to visible light images called Thermal2Sense. The approach is based on the technical routes of diffusion models and Generative Adversarial Networks (GANs) and utilizes a two-stage inference process to optimize the quality of image reconstruction. In the first stage, Thermal2Sense employs a dedicated Controlnet module to reconstruct images with random noise input, ensuring structural consistency with the infrared image. The second stage additionally introduces an IP-Adapter module and Canny Controlnet, leveraging semantic knowledge from pre-trained models for semantic reconstruction at the detail level while maintaining overall structural and color control.

## 🔧 Requirements

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
- cuda==11.8

```bash
conda create --name ConsistentID python=3.8.10
conda activate ConsistentID
pip install -U pip

# Install requirements
pip install -r requirements.txt
```

## 📦️ Data Preparation

Prepare Data in the following format

    ├── data
    |   ├── JSON_all.json 
    |   ├── resize_IMG # Imgaes 
    |   ├── all_faceID  # FaceID
    |   └── parsing_mask_IMG # Parsing Mask 

The .json file should be like
```
[
    {
        "IMG": "Path of image...",
        "parsing_mask_IMG": "...",
        "vqa_llva": "...",
        "id_embed_file_resize": "...",
        "vqa_llva_facial": "..."
    },
    ...
]
```

## 🚀 Train
Ensure that the workspace is the root directory of the project.

```setup
bash train_bash.sh
```

## 🧪 Usage
Ensure that the workspace is the root directory of the project. Then, run [convert_weights.py](https://github.com/JackAILab/ConsistentID/blob/main/evaluation/convert_weights.py) to save the weights efficiently.

### Infer
```setup
python infer.py
```

### Infer Inpaint & Inpaint Controlnet
```setup
python -m demo.inpaint_demo
python -m demo.controlnet_demo
```



## ⏬ Model weights
The model will be automatically downloaded through the following two lines:

```python
from huggingface_hub import hf_hub_download
ConsistentID_path = hf_hub_download(repo_id="JackAILab/ConsistentID", filename="ConsistentID-v1.bin", repo_type="model")
```

The pre-trained model parameters of the model can also be downloaded on [Google Drive](https://drive.google.com/file/d/1jCHICryESmNkzGi8J_FlY3PjJz9gqoSI/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1NAVmH8S7Ls5rZc-snDk1Ng?pwd=nsh6).


## Acknowledgement
* Inspired from many excellent demos and repos, including [IPAdapter](https://github.com/tencent-ailab/IP-Adapter), [FastComposer](https://github.com/mit-han-lab/fastcomposer), [PhotoMaker](https://github.com/TencentARC/PhotoMaker), [InstantID](https://github.com/InstantID/InstantID). Thanks for their great work!
* Thanks to the open source contributions of the following work: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), [LLaVA](https://github.com/haotian-liu/LLaVA), [insightface](https://github.com/deepinsight/insightface), [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA](https://github.com/switchablenorms/CelebAMask-HQ), [SFHQ](https://github.com/SelfishGene/SFHQ-dataset).
* 🤗 Thanks to the huggingface gradio team [ZeroGPUs](https://github.com/huggingface) for their free GPU support!

## Disclaimer
This project strives to impact the domain of AI-driven image generation positively. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.

## Citation
If you found this code helpful, please consider citing:
~~~
@article{huang2024consistentid,
  title={ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving},
  author={Huang, Jiehui and Dong, Xiao and Song, Wenhui and Li, Hanhui and Zhou, Jun and Cheng, Yuhao and Liao, Shutao and Chen, Long and Yan, Yiqiang and Liao, Shengcai and others},
  journal={arXiv preprint arXiv:2404.16771},
  year={2024}
}
~~~


