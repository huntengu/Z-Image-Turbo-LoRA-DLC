import os
import json
import copy
import time
import requests
import random
import logging
import numpy as np
import spaces
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
import gradio as gr

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    ZImagePipeline
)

from huggingface_hub import (
    hf_hub_download,
    HfFileSystem,
    ModelCard,
    snapshot_download)

from diffusers.utils import load_image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red, # Use the new color
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

loras = [
    {
        "image": "https://huggingface.co/Ttio2/Z-Image-Turbo-pencil-sketch/resolve/main/images/z-image_00097_.png",
        "title": "Turbo Pencil",
        "repo": "Ttio2/Z-Image-Turbo-pencil-sketch", #0
        "weights": "Zimage_pencil_sketch.safetensors",
        "trigger_word": "pencil sketch"    
    },
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection/resolve/main/images/1111111111.png",
        "title": "AWPortrait Z",
        "repo": "Shakker-Labs/AWPortrait-Z", #1
        "weights": "AWPortrait-Z.safetensors",
        "trigger_word": "Portrait"    
    },
    {
        "image": "https://huggingface.co/Quorlen/Z-Image-Turbo-Behind-Reeded-Glass-Lora/resolve/main/images/ComfyUI_00391_.png",
        "title": "Behind Reeded Glass",
        "repo": "Quorlen/Z-Image-Turbo-Behind-Reeded-Glass-Lora", #26
        "weights": "Z_Image_Turbo_Behind_Reeded_Glass_Lora_TAV2_000002750.safetensors",
        "trigger_word": "Act1vate!, Behind reeded glass"    
    },
    {
        "image": "https://huggingface.co/ostris/z_image_turbo_childrens_drawings/resolve/main/images/1764433619736__000003000_9.jpg",
        "title": "Childrens Drawings",
        "repo": "ostris/z_image_turbo_childrens_drawings", #2
        "weights": "z_image_turbo_childrens_drawings.safetensors",
        "trigger_word": "Children Drawings"    
    },
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection/resolve/main/images/xcxc.png",
        "title": "Tarot Z",
        "repo": "multimodalart/tarot-z-image-lora", #22
        "weights": "tarot-z-image_000001250.safetensors",
        "trigger_word": "trtcrd"    
    },
    {
        "image": "https://huggingface.co/renderartist/Technically-Color-Z-Image-Turbo/resolve/main/images/ComfyUI_00917_.png",
        "title": "Technically Color Z",
        "repo": "renderartist/Technically-Color-Z-Image-Turbo", #3
        "weights": "Technically_Color_Z_Image_Turbo_v1_renderartist_2000.safetensors",
        "trigger_word": "t3chnic4lly"    
    },
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection/resolve/main/images/z-image_00147_.png",
        "title": "Turbo Ghibli",
        "repo": "Ttio2/Z-Image-Turbo-Ghibli-Style", #19
        "weights": "ghibli_zimage_finetune.safetensors",
        "trigger_word": "Ghibli Style"    
    },
    {
        "image": "https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/images/ComfyUI_00273_.png",
        "title": "Pixel Art",
        "repo": "tarn59/pixel_art_style_lora_z_image_turbo", #4
        "weights": "pixel_art_style_z_image_turbo.safetensors",
        "trigger_word": "Pixel art style."    
    },
    {
        "image": "https://huggingface.co/renderartist/Saturday-Morning-Z-Image-Turbo/resolve/main/images/Saturday_Morning_Z_15.png",
        "title": "Saturday Morning",
        "repo": "renderartist/Saturday-Morning-Z-Image-Turbo", #5
        "weights": "Saturday_Morning_Z_Image_Turbo_v1_renderartist_1250.safetensors",
        "trigger_word": "saturd4ym0rning"    
    },
    {
        "image": "https://huggingface.co/AIImageStudio/ReversalFilmGravure_z_Image_turbo/resolve/main/images/2025-12-01_173047-z_image_z_image_turbo_bf16-435125750859057-euler_10_hires.png",
        "title": "ReversalFilmGravure",
        "repo": "AIImageStudio/ReversalFilmGravure_z_Image_turbo", #6
        "weights": "z_image_turbo_ReversalFilmGravure_v1.0.safetensors",
        "trigger_word": "Reversal Film Gravure, analog film photography"    
    },
    {
        "image": "https://huggingface.co/renderartist/Coloring-Book-Z-Image-Turbo-LoRA/resolve/main/images/CBZ_00274_.png",
        "title": "Coloring Book Z",
        "repo": "renderartist/Coloring-Book-Z-Image-Turbo-LoRA", #7
        "weights": "Coloring_Book_Z_Image_Turbo_v1_renderartist_2000.safetensors",
        "trigger_word": "c0l0ringb00k"    
    },
    {
        "image": "https://huggingface.co/damnthatai/1950s_American_Dream/resolve/main/images/ZImage_20251129163459_135x_00001_.jpg",
        "title": "1950s American Dream",
        "repo": "damnthatai/1950s_American_Dream", #8
        "weights": "5os4m3r1c4n4_z.safetensors",
        "trigger_word": "5os4m3r1c4n4, 1950s, painting, a painting of"    
    },
    {
        "image": "https://huggingface.co/wcde/Z-Image-Turbo-DeJPEG-Lora/resolve/main/images/01.png",
        "title": "DeJPEG",
        "repo": "wcde/Z-Image-Turbo-DeJPEG-Lora", #9
        "weights": "dejpeg_v3.safetensors",
        "trigger_word": ""    
    },
    {
        "image": "https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA/resolve/main/images/n4aSpqa-YFXYo4dtcIg4W.png",
        "title": "DeJPEG",
        "repo": "suayptalha/Z-Image-Turbo-Realism-LoRA", #10
        "weights": "pytorch_lora_weights.safetensors",
        "trigger_word": "Realism"    
    },
    {
        "image": "https://huggingface.co/renderartist/Classic-Painting-Z-Image-Turbo-LoRA/resolve/main/images/Classic_Painting_Z_00247_.png",
        "title": "Classic Painting Z",
        "repo": "renderartist/Classic-Painting-Z-Image-Turbo-LoRA", #11
        "weights": "Classic_Painting_Z_Image_Turbo_v1_renderartist_1750.safetensors",
        "trigger_word": "class1cpa1nt"    
    },
    {
        "image": "https://huggingface.co/DK9/3D_MMORPG_style_z-image-turbo_lora/resolve/main/images/10_with_lora.png",
        "title": "3D MMORPG",
        "repo": "DK9/3D_MMORPG_style_z-image-turbo_lora", #12
        "weights": "lostark_v1.safetensors",
        "trigger_word": ""    
    },
    {
        "image": "https://huggingface.co/Danrisi/Olympus_UltraReal_ZImage/resolve/main/images/Z-Image_01011_.png",
        "title": "Olympus UltraReal",
        "repo": "Danrisi/Olympus_UltraReal_ZImage", #13
        "weights": "Olympus.safetensors",
        "trigger_word": "digital photography, early 2000s compact camera aesthetic, amateur candid shot, digital photography, early 2000s compact camera aesthetic, amateur candid shot, direct flash lighting, hard flash shadow, specular highlights, overexposed highlights"    
    },
    {
        "image": "https://huggingface.co/AiAF/D-ART_Z-Image-Turbo_LoRA/resolve/main/images/example_l3otpwzaz.png",
        "title": "D ART Z Image",
        "repo": "AiAF/D-ART_Z-Image-Turbo_LoRA", #14
        "weights": "D-ART_Z-Image-Turbo.safetensors",
        "trigger_word": "D-ART"    
    },
    {
        "image": "https://huggingface.co/AlekseyCalvin/Marionette_Modernism_Z-image-Turbo_LoRA/resolve/main/bluebirdmandoll.webp",
        "title": "Marionette Modernism",
        "repo": "AlekseyCalvin/Marionette_Modernism_Z-image-Turbo_LoRA", #15
        "weights": "ZImageDadadoll_000003600.safetensors",
        "trigger_word": "DADADOLL style"    
    },
    {
        "image": "https://huggingface.co/AlekseyCalvin/HistoricColor_Z-image-Turbo-LoRA/resolve/main/HSTZgen2.webp",
        "title": "Historic Color Z",
        "repo": "AlekseyCalvin/HistoricColor_Z-image-Turbo-LoRA", #16
        "weights": "ZImage1HST_000004000.safetensors",
        "trigger_word": "HST style"    
    },
    {
        "image": "https://huggingface.co/tarn59/80s_air_brush_style_z_image_turbo/resolve/main/images/ComfyUI_00707_.png",
        "title": "80s Air Brush",
        "repo": "tarn59/80s_air_brush_style_z_image_turbo", #17
        "weights": "80s_air_brush_style_v2_z_image_turbo.safetensors",
        "trigger_word": "80s Air Brush style."    
    },
    {
        "image": "https://huggingface.co/CedarC/Z-Image_360/resolve/main/images/1765505225357__000006750_6.jpg",
        "title": "360panorama",
        "repo": "CedarC/Z-Image_360", #18
        "weights": "Z-Image_360.safetensors",
        "trigger_word": "360panorama"    
    },
    {
        "image": "https://huggingface.co/jj4real/zimage-igbaddie/resolve/main/images/bc73263da73225d636a1677858435a64.jpg",
        "title": "Turbo IG baddie",
        "repo": "jj4real/zimage-igbaddie", #20
        "weights": "zimage-igbaddie.safetensors",
        "trigger_word": "igbaddie"    
    },
    {
        "image": "https://huggingface.co/HAV0X1014/Z-Image-Turbo-KF-Bat-Eared-Fox-LoRA/resolve/main/images/ComfyUI_00132_.png",
        "title": "KF-Bat-Eared",
        "repo": "HAV0X1014/Z-Image-Turbo-KF-Bat-Eared-Fox-LoRA", #21
        "weights": "z-image-turbo-bat_eared_fox.safetensors",
        "trigger_word": "bat_eared_fox_kemono_friends"    
    },
    {
        "image": "https://cdn-uploads.huggingface.co/production/uploads/653cd3049107029eb004f968/IHttgddXu6ZBMo7eyy8p6.png",
        "title": "80s Horror",
        "repo": "neph1/80s_horror_movies_lora_zit", #23
        "weights": "80s_horror_z_80.safetensors",
        "trigger_word": "80s_horror"    
    },
    {
        "image": "https://huggingface.co/Quorlen/z_image_turbo_Sunbleached_Protograph_Style_Lora/resolve/main/images/ComfyUI_00024_.png",
        "title": "Sunbleached Protograph",
        "repo": "Quorlen/z_image_turbo_Sunbleached_Protograph_Style_Lora", #24
        "weights": "zimageturbo_Sunbleach_Photograph_Style_Lora_TAV2_000002750.safetensors",
        "trigger_word": "Act1vate!"    
    },
    {
        "image": "https://huggingface.co/bunnycore/Z-Art-2.1/resolve/main/images/ComfyUI_00069_.png",
        "title": "Z-Art-2.1",
        "repo": "bunnycore/Z-Art-2.1", #25
        "weights": "Z-Image-Art2.1.safetensors",
        "trigger_word": "anime art"    
    },
    {
        "image": "https://huggingface.co/cactusfriend/longfurby-z/resolve/main/images/1764658860954__000003000_1.jpg",
        "title": "Longfurby",
        "repo": "cactusfriend/longfurby-z", #27
        "weights": "longfurbyZ.safetensors",
        "trigger_word": ""    
    },
]

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "Tongyi-MAI/Z-Image-Turbo"

print(f"Loading {base_model} pipeline...")

# Initialize Pipeline
pipe = ZImagePipeline.from_pretrained(
    base_model,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
).to(device)

# ======== AoTI compilation + FA3 ========
# As per reference for optimization
try:
    print("Applying AoTI compilation and FA3...")
    pipe.transformer.layers._repeated_blocks = ["ZImageTransformerBlock"]
    spaces.aoti_blocks_load(pipe.transformer.layers, "zerogpu-aoti/Z-Image", variant="fa3")
    print("Optimization applied successfully.")
except Exception as e:
    print(f"Optimization warning: {e}. Continuing with standard pipeline.")

MAX_SEED = np.iinfo(np.int32).max

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")

def update_selection(evt: gr.SelectData, width, height):
    selected_lora = loras[evt.index]
    new_placeholder = f"Type a prompt for {selected_lora['title']}"
    lora_repo = selected_lora["repo"]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo}) ✅"
    if "aspect" in selected_lora:
        if selected_lora["aspect"] == "portrait":
            width = 768
            height = 1024
        elif selected_lora["aspect"] == "landscape":
            width = 1024
            height = 768
        else:
            width = 1024
            height = 1024
    return (
        gr.update(placeholder=new_placeholder),
        updated_text,
        evt.index,
        width,
        height,
    )

@spaces.GPU
def run_lora(prompt, image_input, image_strength, cfg_scale, steps, selected_index, randomize_seed, seed, width, height, lora_scale, progress=gr.Progress(track_tqdm=True)):
    # Clean up previous LoRAs in both cases
    with calculateDuration("Unloading LoRA"):
        pipe.unload_lora_weights()
    
    # Check if a LoRA is selected
    if selected_index is not None and selected_index < len(loras):
        selected_lora = loras[selected_index]
        lora_path = selected_lora["repo"]
        trigger_word = selected_lora["trigger_word"]
        
        # Prepare Prompt with Trigger Word
        if(trigger_word):
            if "trigger_position" in selected_lora:
                if selected_lora["trigger_position"] == "prepend":
                    prompt_mash = f"{trigger_word} {prompt}"
                else:
                    prompt_mash = f"{prompt} {trigger_word}"
            else:
                prompt_mash = f"{trigger_word} {prompt}"
        else:
            prompt_mash = prompt

        # Load LoRA
        with calculateDuration(f"Loading LoRA weights for {selected_lora['title']}"):
            weight_name = selected_lora.get("weights", None)
            try:
                pipe.load_lora_weights(
                    lora_path, 
                    weight_name=weight_name, 
                    adapter_name="default",
                    low_cpu_mem_usage=True
                )
                # Set adapter scale
                pipe.set_adapters(["default"], adapter_weights=[lora_scale])
            except Exception as e:
                print(f"Error loading LoRA: {e}")
                gr.Warning("Failed to load LoRA weights. Generating with base model.")
    else:
        # Base Model Case
        print("No LoRA selected. Running with Base Model.")
        prompt_mash = prompt
        
    with calculateDuration("Randomizing seed"):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator(device=device).manual_seed(seed)

    # Note: Z-Image-Turbo is strictly T2I in this reference implementation. 
    # Img2Img via image_input is disabled/ignored for this pipeline update.
    
    with calculateDuration("Generating image"):
        # For Turbo models, guidance_scale is typically 0.0
        forced_guidance = 0.0 # Turbo mode
        
        final_image = pipe(
            prompt=prompt_mash,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=forced_guidance,
            generator=generator,
        ).images[0]
        
    yield final_image, seed, gr.update(visible=False)

def get_huggingface_safetensors(link):
  split_link = link.split("/")
  if(len(split_link) == 2):
            model_card = ModelCard.load(link)
            base_model = model_card.data.get("base_model")
            print(base_model)
      
            # Relaxed check to allow Z-Image or Flux or others, assuming user knows what they are doing
            # or specifically check for Z-Image-Turbo
            if base_model not in ["Tongyi-MAI/Z-Image-Turbo", "black-forest-labs/FLUX.1-dev"]:
                # Just a warning instead of error to allow experimentation
                print("Warning: Base model might not match.")
                
            image_path = model_card.data.get("widget", [{}])[0].get("output", {}).get("url", None)
            trigger_word = model_card.data.get("instance_prompt", "")
            image_url = f"https://huggingface.co/{link}/resolve/main/{image_path}" if image_path else None
            fs = HfFileSystem()
            try:
                list_of_files = fs.ls(link, detail=False)
                for file in list_of_files:
                    if(file.endswith(".safetensors")):
                        safetensors_name = file.split("/")[-1]
                    if (not image_url and file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                      image_elements = file.split("/")
                      image_url = f"https://huggingface.co/{link}/resolve/main/{image_elements[-1]}"
            except Exception as e:
              print(e)
              gr.Warning(f"You didn't include a link neither a valid Hugging Face repository with a *.safetensors LoRA")
              raise Exception(f"You didn't include a link neither a valid Hugging Face repository with a *.safetensors LoRA")
            return split_link[1], link, safetensors_name, trigger_word, image_url

def check_custom_model(link):
    if(link.startswith("https://")):
        if(link.startswith("https://huggingface.co") or link.startswith("https://www.huggingface.co")):
            link_split = link.split("huggingface.co/")
            return get_huggingface_safetensors(link_split[1])
    else: 
        return get_huggingface_safetensors(link)

def add_custom_lora(custom_lora):
    global loras
    if(custom_lora):
        try:
            title, repo, path, trigger_word, image = check_custom_model(custom_lora)
            print(f"Loaded custom LoRA: {repo}")
            card = f'''
            <div class="custom_lora_card">
              <span>Loaded custom LoRA:</span>
              <div class="card_internal">
                <img src="{image}" />
                <div>
                    <h3>{title}</h3>
                    <small>{"Using: <code><b>"+trigger_word+"</code></b> as the trigger word" if trigger_word else "No trigger word found. If there's a trigger word, include it in your prompt"}<br></small>
                </div>
              </div>
            </div>
            '''
            existing_item_index = next((index for (index, item) in enumerate(loras) if item['repo'] == repo), None)
            if(not existing_item_index):
                new_item = {
                    "image": image,
                    "title": title,
                    "repo": repo,
                    "weights": path,
                    "trigger_word": trigger_word
                }
                print(new_item)
                existing_item_index = len(loras)
                loras.append(new_item)
        
            return gr.update(visible=True, value=card), gr.update(visible=True), gr.Gallery(selected_index=None), f"Custom: {path}", existing_item_index, trigger_word
        except Exception as e:
            gr.Warning(f"Invalid LoRA: either you entered an invalid link, or a non-supported LoRA")
            return gr.update(visible=True, value=f"Invalid LoRA: either you entered an invalid link, a non-supported LoRA"), gr.update(visible=False), gr.update(), "", None, ""
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

def remove_custom_lora():
    return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

run_lora.zerogpu = True

css = '''
#gen_btn{height: 100%}
#gen_column{align-self: stretch}
#title{text-align: center}
#title h1{font-size: 3em; display:inline-flex; align-items:center}
#title img{width: 100px; margin-right: 0.5em}
#gallery .grid-wrap{height: 10vh}
#lora_list{background: var(--block-background-fill);padding: 0 1em .3em; font-size: 90%}
.card_internal{display: flex;height: 100px;margin-top: .5em}
.card_internal img{margin-right: 1em}
.styler{--form-gap-width: 0px !important}
#progress{height:30px}
#progress .generating{display:none}
.progress-container {width: 100%;height: 30px;background-color: #f0f0f0;border-radius: 15px;overflow: hidden;margin-bottom: 20px}
.progress-bar {height: 100%;background-color: #4f46e5;width: calc(var(--current) / var(--total) * 100%);transition: width 0.5s ease-in-out}
'''

with gr.Blocks(delete_cache=(60, 60)) as demo:
    title = gr.HTML(
        """<h1>Z Image Turbo LoRA DLC 🧪</h1>""",
        elem_id="title",
    )
    selected_index = gr.State(None)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Enter Prompt", lines=1, placeholder="✦︎ Choose the LoRA and type the prompt (LoRA = None → Base Model = Active)")
        with gr.Column(scale=1, elem_id="gen_column"):
            generate_button = gr.Button("Generate", variant="primary", elem_id="gen_btn")
    with gr.Row():
        with gr.Column():
            selected_info = gr.Markdown("### No LoRA Selected (Base Model)")
            gallery = gr.Gallery(
                [(item["image"], item["title"]) for item in loras],
                label="Z-Image LoRAs",
                allow_preview=False,
                columns=3,
                elem_id="gallery",
            )
            with gr.Group():
                custom_lora = gr.Textbox(label="Enter Custom LoRA", placeholder="Paste the LoRA path and press Enter (e.g., Shakker-Labs/AWPortrait-Z).")
                gr.Markdown("[Check the list of Z-Image LoRA's](https://huggingface.co/models?other=base_model:adapter:Tongyi-MAI/Z-Image-Turbo)", elem_id="lora_list")
            custom_lora_info = gr.HTML(visible=False)
            custom_lora_button = gr.Button("Remove custom LoRA", visible=False)
        with gr.Column():
            progress_bar = gr.Markdown(elem_id="progress",visible=False)
            result = gr.Image(label="Generated Image", format="png", height=630)

    with gr.Row():
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                input_image = gr.Image(label="Input image (Ignored for Z-Image-Turbo)", type="filepath", visible=False)
                image_strength = gr.Slider(label="Denoise Strength", info="Ignored for Z-Image-Turbo", minimum=0.1, maximum=1.0, step=0.01, value=0.75, visible=False)
            with gr.Column():
                with gr.Row():
                    cfg_scale = gr.Slider(label="CFG Scale", info="Forced to 0.0 for Turbo", minimum=0, maximum=20, step=0.5, value=0.0, interactive=False)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=9)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1536, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=1536, step=64, value=1024)
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)
                    lora_scale = gr.Slider(label="LoRA Scale", minimum=0, maximum=3, step=0.01, value=0.95)

    gallery.select(
        update_selection,
        inputs=[width, height],
        outputs=[prompt, selected_info, selected_index, width, height]
    )
    custom_lora.input(
        add_custom_lora,
        inputs=[custom_lora],
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, prompt]
    )
    custom_lora_button.click(
        remove_custom_lora,
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, custom_lora]
    )
    gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn=run_lora,
        inputs=[prompt, input_image, image_strength, cfg_scale, steps, selected_index, randomize_seed, seed, width, height, lora_scale],
        outputs=[result, seed, progress_bar]
    )

demo.queue()
demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)
