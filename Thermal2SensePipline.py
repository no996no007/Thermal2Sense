import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2
import PIL
import numpy as np 
from PIL import Image
import torch
from safetensors import safe_open
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers import StableDiffusionControlNetPipeline,ControlNetModel,LMSDiscreteScheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from ip_adapter.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class T2VStableDiffusionPipeline(StableDiffusionControlNetPipeline):
    def load_IPA(self, image_encoder_path, ip_ckpt, num_tokens=4):
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self._execution_device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self._execution_device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self._execution_device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self, "controlnet"):
            if isinstance(self.controlnet, MultiControlNetModel):
                for controlnet in self.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values #torch.Size([1, 3, 224, 224])
            clip_image_embeds = self.image_encoder(clip_image.to(self._execution_device, dtype=torch.float16)).image_embeds #torch.Size([1, 1024])
        else:
            clip_image_embeds = clip_image_embeds.to(self._execution_device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds) #torch.Size([1, 4, 768])
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
    ):
        """
        调用函数进行图像生成。

        参数:
            prompt (`str` 或 `List[str]`, *可选*):
                用于引导图像生成的提示词或提示词列表。如果未定义，则需要传递 `prompt_embeds`。
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`, 
                `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` 或 `List[List[PIL.Image.Image]]`):
                提供给 ControlNet 的输入条件，用于引导 `unet` 生成图像。如果类型指定为 `torch.FloatTensor`，则按原样传递给 ControlNet。
                `PIL.Image.Image` 也可以作为图像接受。输出图像的尺寸默认为 `image` 的尺寸。如果传递了高度和/或宽度，则根据需要调整 `image` 的大小。
                如果在 `init` 中指定了多个 ControlNet，则必须将图像作为列表传递，以便每个列表元素可以正确批处理为单个 ControlNet 的输入。
            height (`int`, *可选*, 默认值为 `self.unet.config.sample_size * self.vae_scale_factor`):
                生成图像的高度（以像素为单位）。
            width (`int`, *可选*, 默认值为 `self.unet.config.sample_size * self.vae_scale_factor`):
                生成图像的宽度（以像素为单位）。
            num_inference_steps (`int`, *可选*, 默认值为 50):
                去噪步骤的数量。更多的去噪步骤通常会导致更高质量的图像，但推理速度较慢。
            guidance_scale (`float`, *可选*, 默认值为 7.5):
                较高的指导尺度值会鼓励模型生成与文本 `prompt` 密切相关的图像，但图像质量较低。仅当 `guidance_scale > 1` 时启用指导尺度。
            negative_prompt (`str` 或 `List[str]`, *可选*):
                用于引导图像生成中不包含的内容的提示词或提示词列表。如果未定义，则需要传递 `negative_prompt_embeds`。当不使用指导（`guidance_scale < 1`）时忽略。
            num_images_per_prompt (`int`, *可选*, 默认值为 1):
                每个提示词生成的图像数量。
            eta (`float`, *可选*, 默认值为 0.0):
                对应于 [DDIM](https://arxiv.org/abs/2010.02502) 论文中的参数 eta (η)。仅适用于 [`~schedulers.DDIMScheduler`]，在其他调度器中忽略。
            generator (`torch.Generator` 或 `List[torch.Generator]`, *可选*):
                用于生成确定性的 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)。
            latents (`torch.FloatTensor`, *可选*):
                预生成的从高斯分布中采样的噪声潜变量，用作图像生成的输入。可以使用不同的提示词调整相同的生成。如果未提供，则通过使用提供的随机 `generator` 采样生成潜变量张量。
            prompt_embeds (`torch.FloatTensor`, *可选*):
                预生成的文本嵌入。可以轻松调整文本输入（提示词加权）。如果未提供，则从 `prompt` 输入参数生成文本嵌入。
            negative_prompt_embeds (`torch.FloatTensor`, *可选*):
                预生成的负文本嵌入。可以轻松调整文本输入（提示词加权）。如果未提供，则从 `negative_prompt` 输入参数生成负文本嵌入。
            output_type (`str`, *可选*, 默认值为 `"pil"`):
                生成图像的输出格式。选择 `PIL.Image` 或 `np.array`。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] 而不是普通元组。
            callback (`Callable`, *可选*):
                在推理过程中每 `callback_steps` 步调用的函数。该函数调用时带有以下参数：`callback(step: int, timestep: int, latents: torch.FloatTensor)`。
            callback_steps (`int`, *可选*, 默认值为 1):
                调用 `callback` 函数的频率。如果未指定，则每一步调用一次回调。
            cross_attention_kwargs (`dict`, *可选*):
                如果指定，则将该 kwargs 字典传递给 [`AttentionProcessor`]，如 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 中定义。
            controlnet_conditioning_scale (`float` 或 `List[float]`, *可选*, 默认值为 1.0):
                在将 ControlNet 的输出添加到原始 `unet` 的残差之前，将其乘以 `controlnet_conditioning_scale`。如果在 `init` 中指定了多个 ControlNet，可以将相应的比例设置为列表。
            guess_mode (`bool`, *可选*, 默认值为 `False`):
                ControlNet 编码器即使移除所有提示词也会尝试识别输入图像的内容。推荐的 `guidance_scale` 值在 3.0 到 5.0 之间。
            control_guidance_start (`float` 或 `List[float]`, *可选*, 默认值为 0.0):
                ControlNet 开始应用的总步骤百分比。
            control_guidance_end (`float` 或 `List[float]`, *可选*, 默认值为 1.0):
                ControlNet 停止应用的总步骤百分比。
            clip_skip (`int`, *可选*):
                计算提示词嵌入时要跳过的 CLIP 层数。值为 1 表示将使用倒数第二层的输出来计算提示词嵌入。

        示例:

        返回值:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] 或 `tuple`:
                如果 `return_dict` 为 `True`，则返回 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]，否则返回元组，其中第一个元素是生成的图像列表，第二个元素是 `bool` 列表，指示相应生成的图像是否包含“不安全内容” (nsfw)。
        """
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad() 
    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None, # 必要的参数
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        start_step: int = 10, # 唯一增加的参数
    ):
        """
        调用函数进行图像生成。

        参数:
            prompt (`str` 或 `List[str]`, *可选*):
                用于引导图像生成的提示词或提示词列表。如果未定义，则需要传递 `prompt_embeds`。
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`, 
                `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` 或 `List[List[PIL.Image.Image]]`):
                提供给 ControlNet 的输入条件，用于引导 `unet` 生成图像。如果类型指定为 `torch.FloatTensor`，则按原样传递给 ControlNet。
                `PIL.Image.Image` 也可以作为图像接受。输出图像的尺寸默认为 `image` 的尺寸。如果传递了高度和/或宽度，则根据需要调整 `image` 的大小。
                如果在 `init` 中指定了多个 ControlNet，则必须将图像作为列表传递，以便每个列表元素可以正确批处理为单个 ControlNet 的输入。
            height (`int`, *可选*, 默认值为 `self.unet.config.sample_size * self.vae_scale_factor`):
                生成图像的高度（以像素为单位）。
            width (`int`, *可选*, 默认值为 `self.unet.config.sample_size * self.vae_scale_factor`):
                生成图像的宽度（以像素为单位）。
            num_inference_steps (`int`, *可选*, 默认值为 50):
                去噪步骤的数量。更多的去噪步骤通常会导致更高质量的图像，但推理速度较慢。
            guidance_scale (`float`, *可选*, 默认值为 7.5):
                较高的指导尺度值会鼓励模型生成与文本 `prompt` 密切相关的图像，但图像质量较低。仅当 `guidance_scale > 1` 时启用指导尺度。
            negative_prompt (`str` 或 `List[str]`, *可选*):
                用于引导图像生成中不包含的内容的提示词或提示词列表。如果未定义，则需要传递 `negative_prompt_embeds`。当不使用指导（`guidance_scale < 1`）时忽略。
            num_images_per_prompt (`int`, *可选*, 默认值为 1):
                每个提示词生成的图像数量。
            eta (`float`, *可选*, 默认值为 0.0):
                对应于 [DDIM](https://arxiv.org/abs/2010.02502) 论文中的参数 eta (η)。仅适用于 [`~schedulers.DDIMScheduler`]，在其他调度器中忽略。
            generator (`torch.Generator` 或 `List[torch.Generator]`, *可选*):
                用于生成确定性的 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)。
            latents (`torch.FloatTensor`, *可选*):
                预生成的从高斯分布中采样的噪声潜变量，用作图像生成的输入。可以使用不同的提示词调整相同的生成。如果未提供，则通过使用提供的随机 `generator` 采样生成潜变量张量。
            prompt_embeds (`torch.FloatTensor`, *可选*):
                预生成的文本嵌入。可以轻松调整文本输入（提示词加权）。如果未提供，则从 `prompt` 输入参数生成文本嵌入。
            negative_prompt_embeds (`torch.FloatTensor`, *可选*):
                预生成的负文本嵌入。可以轻松调整文本输入（提示词加权）。如果未提供，则从 `negative_prompt` 输入参数生成负文本嵌入。
            output_type (`str`, *可选*, 默认值为 `"pil"`):
                生成图像的输出格式。选择 `PIL.Image` 或 `np.array`。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] 而不是普通元组。
            callback (`Callable`, *可选*):
                在推理过程中每 `callback_steps` 步调用的函数。该函数调用时带有以下参数：`callback(step: int, timestep: int, latents: torch.FloatTensor)`。
            callback_steps (`int`, *可选*, 默认值为 1):
                调用 `callback` 函数的频率。如果未指定，则每一步调用一次回调。
            cross_attention_kwargs (`dict`, *可选*):
                如果指定，则将该 kwargs 字典传递给 [`AttentionProcessor`]，如 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 中定义。
            controlnet_conditioning_scale (`float` 或 `List[float]`, *可选*, 默认值为 1.0):
                在将 ControlNet 的输出添加到原始 `unet` 的残差之前，将其乘以 `controlnet_conditioning_scale`。如果在 `init` 中指定了多个 ControlNet，可以将相应的比例设置为列表。
            guess_mode (`bool`, *可选*, 默认值为 `False`):
                ControlNet 编码器即使移除所有提示词也会尝试识别输入图像的内容。推荐的 `guidance_scale` 值在 3.0 到 5.0 之间。
            control_guidance_start (`float` 或 `List[float]`, *可选*, 默认值为 0.0):
                ControlNet 开始应用的总步骤百分比。
            control_guidance_end (`float` 或 `List[float]`, *可选*, 默认值为 1.0):
                ControlNet 停止应用的总步骤百分比。
            clip_skip (`int`, *可选*):
                计算提示词嵌入时要跳过的 CLIP 层数。值为 1 表示将使用倒数第二层的输出来计算提示词嵌入。
            start_step:
                从输入的噪声版本开始的循环（又名image2image）当我们使用图像作为起点，添加一些噪声，在循环中执行最后几个去噪步骤时会发生什么。将跳过第一个“start_step”步骤。
                代码会使用调度器将其噪声处理到相当于步骤10的级别（“start_step”）。

        示例:

        返回值:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] 或 `tuple`:
                如果 `return_dict` 为 `True`，则返回 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]，否则返回元组，其中第一个元素是生成的图像列表，第二个元素是 `bool` 列表，指示相应生成的图像是否包含“不安全内容” (nsfw)。
        """
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt( #both torch.Size([1, 77, 768])
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        # self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.scheduler.set_timesteps(num_inference_steps,device=device)
        # self.scheduler.timesteps = self.scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925
        timesteps = self.scheduler.timesteps

        # add noise to latents input to the level of step of "start_step"
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps=torch.tensor([self.scheduler.timesteps[start_step]]))
        # latents = latents.to(device).float()
        # # 按照scheduler所需的标准差来缩放初始噪声
        # latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i >= start_step: # << This is the only modification to the loop we do
                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # controlnet(s) inference
                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
    def generate_stage1(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds( #both torch.Size([1, 4, 768])
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1) #torch.Size([1, 4->16, 768]) 处理同时输出4张图片的情况，如果是一张那就torch.Size([1, 4->4, 768])
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1) #torch.Size([1->4, 16->4, 768])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.encode_prompt( #both torch.Size([4, 77, 768])
                prompt,
                device=self._execution_device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1) #torch.Size([4, 81, 768])
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1) #torch.Size([4, 81, 768])

        images = self(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **kwargs,
        ).images

        return images

    def generate_stage2(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds( #both torch.Size([1, 4, 768])
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1) #torch.Size([1, 4->16, 768]) 处理同时输出4张图片的情况，如果是一张那就torch.Size([1, 4->4, 768])
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1) #torch.Size([1->4, 16->4, 768])
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.encode_prompt( #both torch.Size([4, 77, 768])
                prompt,
                device=self._execution_device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1) #torch.Size([4, 81, 768])
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1) #torch.Size([4, 81, 768])

        images = self.img2img(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **kwargs,
        ).images

        return images


    @torch.no_grad()
    def latents_to_pil(self,latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @torch.no_grad()
    def pil_to_latent(self,input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = self.vae.encode(transforms.ToTensor()(input_im).unsqueeze(0).to(self._execution_device)*2-1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()