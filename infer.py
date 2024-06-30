import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import numpy as np
import cv2
from diffusers import ControlNetModel
from diffusers.utils import load_image
import torch

# private lib
from Thermal2SensePipline import T2VStableDiffusionPipeline

def dynamic_canny_edge_detection(image, sigma=0.33):
    image_np = np.array(image.convert('L'))  # 转换为灰度图像
    # 计算中位值
    v = np.median(image_np)
    # 根据中位值计算动态阈值
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # 使用 OpenCV 进行 Canny 边缘检测
    edges = cv2.Canny(image_np, lower, upper)
    # 将结果转换为 RGB 图像
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # 将结果转换为 PIL 图像对象
    edges_image = Image.fromarray(edges_rgb)
    return edges_image

# 加载diffuser格式，分开记载基础模型与Controlnet模型
device = "cuda"
base_model_path = "/data6/huangjiehui_m22/pretrained_model/Realistic_Vision_V6.0_B1_noVAE"
controlnet_model_path_1 = "/data6/huangjiehui_m22/z_benke/liaost/T2V_on_SD/models/T2V_controlnet"
controlnet_model_path_2 = "/data6/huangjiehui_m22/z_benke/liaost/T2V_on_SD/models/canny-controlnet"

controlnet_1 = ControlNetModel.from_pretrained(controlnet_model_path_1, torch_dtype=torch.float16)
controlnet_2 = ControlNetModel.from_pretrained(controlnet_model_path_2, torch_dtype=torch.float16)

pipe = T2VStableDiffusionPipeline.from_pretrained(
    base_model_path,
    controlnet=[controlnet_1,controlnet_2],
    torch_dtype=torch.float16,
).to(device)

image_encoder_path = "/data6/huangjiehui_m22/pretrained_model/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "/data6/huangjiehui_m22/z_benke/liaost/T2V_on_SD/models/T2V_ipadapter/ip_adapter24000.bin" 


pipe.load_IPA(
    image_encoder_path,
    ip_ckpt,
)

# hyper-parameter
num_steps = 50
merge_steps = 50
seed = torch.randint(0, 1000, (1,)).item()
generator = torch.Generator(device="cuda").manual_seed(seed)

T2Vcontrolnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/train_512x512_100/TH/set01V000lwirI00531.jpg") #Input

# 第一次推理，生成 latents
latents = pipe.generate_stage1(
    prompt="",
    pil_image=Image.new('RGB', T2Vcontrolnet_image.size, (0, 0, 0)),
    scale=0.0,
    width=512,
    height=512,
    num_inference_steps=num_steps,
    generator=generator,
    image=[T2Vcontrolnet_image,Image.new('RGB', T2Vcontrolnet_image.size, (0, 0, 0))],
    output_type = "latent",
    guidance_scale=2,
    controlnet_conditioning_scale=[1.0,0],
)

# 第二次推理，不使用 Controlnet，做img2img
images = pipe.generate_stage2(
    prompt="",
    pil_image=pipe.latents_to_pil(latents)[0],
    scale=0.3,
    width=512,
    height=512,
    num_inference_steps=num_steps,
    generator=generator,
    latents=latents,
    image=[T2Vcontrolnet_image,dynamic_canny_edge_detection(pipe.latents_to_pil(latents)[0])],
    guidance_scale=4,
    guess_mode = True,
    controlnet_conditioning_scale=[0.4,0.5], #再大一点就严重干扰文本的语义了，绘制不了语义完整的物体
    start_step = 10, # 重要参数，加噪到第几步，也是开始denoise的步骤
)[0]

# Gets the absolute path of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))
images.save(script_directory+f"/outputs/sample {seed}.jpg")
print("Save to: ","script_directory"+f"/outputs/sample {seed}.jpg")
