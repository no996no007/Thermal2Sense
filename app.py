import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import numpy as np
import cv2
from diffusers import ControlNetModel
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
import gradio as gr
import torch
from PIL import Image
import os
os.environ['GRADIO_TEMP_DIR'] = "/data6/huangjiehui_m22/z_benke/liaost/ConsistentID/images/gradio_tmp" #这一个非常重要，不设置gradio公网运行不了！

# 定义超参数
num_steps = 50
merge_steps = 50
seed = torch.randint(0, 1000, (1,)).item()
stage2_ipa_scale = 0.3
stage2_guidance_scale = 4
stage2_T2V_controlnat_scale = 0.4
stage2_canny_controlnat_scale = 0.5
stage2_start_step = 10

# 模板图像文件夹路径
template_dir = "./templates"
preset_template = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))]

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# 生成图像的函数
def generate_image(num_steps, seed, stage2_ipa_scale, stage2_guidance_scale, stage2_T2V_controlnat_scale, stage2_canny_controlnat_scale, stage2_start_step, input_image_path=None, upload_image=None):
    if input_image_path:
        T2Vcontrolnet_image = load_image(input_image_path)
    elif upload_image:
        T2Vcontrolnet_image = Image.open(upload_image).convert("RGB")
    else:
        return None

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 第一次推理
    latents = pipe.generate_stage1(
        prompt="",
        pil_image=Image.new('RGB', T2Vcontrolnet_image.size, (0, 0, 0)),
        scale=0.0,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        generator=generator,
        image=[T2Vcontrolnet_image, Image.new('RGB', T2Vcontrolnet_image.size, (0, 0, 0))],
        output_type="latent",
        guidance_scale=2,
        controlnet_conditioning_scale=[1.0, 0],
    )

    # 第二次推理
    images = pipe.generate_stage2(
        prompt="",
        pil_image=pipe.latents_to_pil(latents)[0],
        scale=stage2_ipa_scale,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        generator=generator,
        latents=latents,
        image=[T2Vcontrolnet_image, dynamic_canny_edge_detection(pipe.latents_to_pil(latents)[0])],
        guidance_scale=stage2_guidance_scale,
        guess_mode=True,
        controlnet_conditioning_scale=[stage2_T2V_controlnat_scale, stage2_canny_controlnat_scale],
        start_step=stage2_start_step,
    )[0]

    # 保存图像
    script_directory = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(script_directory, f"outputs/sample_{seed}.jpg")
    images.save(output_path)
    return output_path

# 使用Blocks创建Gradio界面
with gr.Blocks(title="ConsistentID Demo") as demo:
    with gr.Row():
        with gr.Column():
            model_selected_tab = gr.State(0)
            with gr.TabItem("Template Images") as template_images_tab:
                template_gallery_list = [(load_image(i), i) for i in preset_template]
                gallery = gr.Gallery(template_gallery_list, columns=4, rows=2, object_fit="contain", height="auto", show_label=False)
                
                def select_function(evt: gr.SelectData):
                    return preset_template[evt.index]

                selected_template_images = gr.Text(show_label=False, visible=False, placeholder="Selected")
                gallery.select(select_function, None, selected_template_images)
            with gr.TabItem("Upload Image") as upload_image_tab:
                costum_image = gr.Image(label="Upload Image")

            model_selected_tabs = [template_images_tab, upload_image_tab]
            for i, tab in enumerate(model_selected_tabs):
                tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[model_selected_tab])

        with gr.Column():
            num_steps_slider = gr.Slider(minimum=10, maximum=100, step=1, value=num_steps, label="Number of Steps")
            seed_slider = gr.Slider(minimum=0, maximum=1000, step=1, value=seed, label="Seed")
            stage2_ipa_scale_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=stage2_ipa_scale, label="Stage 2 IPA Scale")
            stage2_guidance_scale_slider = gr.Slider(minimum=1, maximum=10, step=1, value=stage2_guidance_scale, label="Stage 2 Guidance Scale")
            stage2_T2V_controlnat_scale_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=stage2_T2V_controlnat_scale, label="Stage 2 T2V ControlNet Scale")
            stage2_canny_controlnat_scale_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=stage2_canny_controlnat_scale, label="Stage 2 Canny ControlNet Scale")
            stage2_start_step_slider = gr.Slider(minimum=0, maximum=20, step=1, value=stage2_start_step, label="Stage 2 Start Step")

            output_image = gr.Image(label="Generated Image")

            generate_button = gr.Button("Generate Image")

            generate_button.click(
                fn=generate_image,
                inputs=[
                    num_steps_slider,
                    seed_slider,
                    stage2_ipa_scale_slider,
                    stage2_guidance_scale_slider,
                    stage2_T2V_controlnat_scale_slider,
                    stage2_canny_controlnat_scale_slider,
                    stage2_start_step_slider,
                    selected_template_images,
                    costum_image,
                ],
                outputs=output_image,
            )

demo.launch()
