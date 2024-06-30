import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import numpy as np
import cv2
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_controlnet_checkpoint,download_from_original_stable_diffusion_ckpt
from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers.utils import load_image
import yaml
import torch
import random
import shutil

# private lib
from pipline_controlnet_T2V import T2VStableDiffusionPipeline

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
# ip_ckpt = "/data6/huangjiehui_m22/pretrained_model/IP-Adapter/models/ip-adapter_sd15.bin"
# ip_ckpt = "/data6/huangjiehui_m22/z_benke/liaost/T2V_on_SD/models/T2V_ipadapter/ip_adapter108000.bin"
ip_ckpt = "/data6/huangjiehui_m22/z_benke/liaost/T2V_on_SD/models/T2V_ipadapter/ip_adapter24000.bin"


pipe.load_IPA(
    image_encoder_path,
    ip_ckpt,
)


# 源文件夹，包含多张照片
TH_source_folder = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test/TH"
VIS_source_folder = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test/VIS"
TH_source_folder = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/TH"
VIS_source_folder = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/VIS"
# 目标文件夹，用于存放选中的照片
destination_folder_pred = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/results/PRED"
destination_folder_pred_gt = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/results/PRED-GT"
destination_folder_pred_input = "/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/results/PRED-INPUT"

# 要选择的照片数量
num_images_to_select = 50

# 获取源文件夹中所有照片的文件名列表
TH_image_files = [filename for filename in os.listdir(TH_source_folder) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
TH_image_files.sort()
VIS_image_files = [filename for filename in os.listdir(VIS_source_folder) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
VIS_image_files.sort()

# 随机选择 n 张照片
random.seed(955)
random.seed(996)
random_numbers = [random.randint(0, len(TH_image_files)-1) for _ in range(num_images_to_select)]

# 将选中的原照片剪切到目标文件夹
for i in random_numbers:
    TH_filename = TH_image_files[i] # 粘贴过去后名字要保持一致，都为TH的名字
    VIS_filename = VIS_image_files[i] 
    shutil.copy(os.path.join(TH_source_folder, TH_filename), os.path.join(destination_folder_pred_input, TH_filename))  # 使用 shutil.move() 进行剪切操作
    shutil.copy(os.path.join(VIS_source_folder, VIS_filename), os.path.join(destination_folder_pred_gt, TH_filename))  # 使用 shutil.move() 进行剪切操作
    # 预测图像地址
    controlnet_image = load_image(os.path.join(TH_source_folder, TH_filename))

    # hyper-parameter
    num_steps = 50
    merge_steps = 50
    prompt = "Street view cars roads Busy streets, intersections There are buildings, signs, and command lights in the distance"
    prompt1 = ""
    prompt2 = "The dim road at night, with only streetlights and a few cars, is lined with pitch black small trees on both sides of the road"
    prompt3 = "Possible streetscape, daytime, dim night, high-rise buildings, traffic lights, pedestrians, green belts, trees, grass, roadside, seaside, boats, houses, towers"

    seed = torch.randint(0, 1000, (1,)).item()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/TH/JPEGImagestestirFLIR_08928_PreviewData.jpg")
    # controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/TH/JPEGImagestrainirFLIR_00300_PreviewData.jpg")#1
    # controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/TH/JPEGImagestrainirFLIR_00381_PreviewData.jpg")#2
    # controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/TH/LLVIPinfraredtest190051.jpg") #3
    # controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512_3samples/VIS/JPEGImagestestrgbFLIR_09057_RGB.jpg") #3
    # controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/test_512x512/TH/M3FDM3FD_DetectionIr04090.jpg") #4
    controlnet_image = load_image("/data6/huangjiehui_m22/z_benke/liaost/datasets/T2V/results/PRED-INPUT/866.png") #4
    
    
    # controlnet_image = controlnet_image.resize((256, 256))
    # from PIL import Image, ImageFilter
    # controlnet_image = controlnet_image.filter(ImageFilter.GaussianBlur(radius=2))
    # controlnet_image = controlnet_image.resize((512, 512))

    # 第一次推理，生成 latents
    latents = pipe.generate_stage1(
        prompt="",
        pil_image=Image.new('RGB', controlnet_image.size, (0, 0, 0)),
        scale=0.0,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        generator=generator,
        image=[controlnet_image,Image.new('RGB', controlnet_image.size, (0, 0, 0))],
        output_type = "latent",
        guidance_scale=2,
        # guess_mode = True,
        controlnet_conditioning_scale=[1.0,0],
    )

    # 第二次推理，不使用 Controlnet，做img2img
    images = pipe.generate_stage2(
        prompt="",
        pil_image=pipe.latents_to_pil(latents)[0],
        # pil_image=load_image(os.path.join(VIS_source_folder, VIS_filename)),
        scale=0.3,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        generator=generator,
        latents=latents,
        image=[controlnet_image,dynamic_canny_edge_detection(pipe.latents_to_pil(latents)[0])],
        guidance_scale=4,
        guess_mode = True,
        controlnet_conditioning_scale=[0.4,0.5], #再大一点就严重干扰文本的语义了，绘制不了语义完整的物体
        start_step = 10, # 重要参数，加噪到第几步，也是开始denoise的步骤
    )[0]

    # Gets the absolute path of the current script
    script_directory = os.path.dirname(os.path.realpath(__file__))
    pipe.latents_to_pil(latents)[0].save(script_directory+f"/images/test_outputs4/sample {seed} stage1.jpg")
    print("Save to: ","script_directory"+f"/images/test_outputs4/sample {seed} stage1.jpg")
    images.save(script_directory+f"/images/test_outputs4/sample {seed}.jpg")
    print("Save to: ","script_directory"+f"/images/test_outputs4/sample {seed}.jpg")


    # # 第一次推理，生成 latents
    # latents = pipe(
    #     prompt=prompt1,
    #     width=512,
    #     height=512,
    #     num_inference_steps=num_steps,
    #     generator=generator,
    #     image=[controlnet_image,Image.new('RGB', controlnet_image.size, (0, 0, 0))],
    #     output_type = "latent",
    #     guidance_scale=2,
    #     controlnet_conditioning_scale=[1.0,0],
    # ).images

    # # 第二次推理，不使用 Controlnet，做img2img
    # images = pipe.img2img(
    #     prompt=prompt2,
    #     width=512,
    #     height=512,
    #     num_inference_steps=num_steps,
    #     generator=generator,
    #     latents=latents,
    #     image=[controlnet_image,dynamic_canny_edge_detection(pipe.latents_to_pil(latents)[0])],
    #     guidance_scale=2,
    #     controlnet_conditioning_scale=[0.3,0.5], #再大一点就严重干扰文本的语义了，绘制不了语义完整的物体
    #     start_step = 10, # 重要参数，加噪到第几步，也是开始denoise的步骤
    # ).images[0]