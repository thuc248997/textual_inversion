import os
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

import PIL
import torch
from torch.nn.functional import cosine_similarity
from torchvision import transforms
import numpy as np
import clip

def generate_image(
    pipe,
    init_seed,
    num_image,
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    name="exp",
):
    os.makedirs(name, exist_ok=True)
    for i in range(num_image):
        seed = init_seed + i
        torch.manual_seed(seed)
        generator.manual_seed(seed)
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.save("{}/{}.png".format(name, seed))

class CLIPEvaluator:
    def __init__(self, device, clip_model="ViT-B/32") -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
            + clip_preprocess.transforms[:2]
            + clip_preprocess.transforms[4:]  # to match CLIP input scale assumptions
        )  # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def get_image_features(self, images) -> torch.Tensor:
        if isinstance(images[0], PIL.Image.Image):
            # images is a list of PIL Images
            images = torch.stack([self.clip_preprocess(image) for image in images]).to(
                self.device
            )
        else:
            # images is a tensor of [-1, 1] images
            images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens)
        return text_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)
        if src_img_features.shape[0] == gen_img_features.shape[0]:
            return cosine_similarity(src_img_features, gen_img_features).mean().item()
        else:
            scores = []
            for idx in range(src_img_features.shape[0]):
                src_img_feature = src_img_features[idx].unsqueeze(0)
                scores.append(
                    cosine_similarity(src_img_feature, gen_img_features).mean().item()
                )
            return np.mean(scores)

    def txt_to_img_similarity(self, text, generated_images, reduction=True):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        if reduction:
            return cosine_similarity(text_features, gen_img_features).mean().item()
        else:
            return cosine_similarity(text_features, gen_img_features)


def read_dir_image(path):
    images = []
    for img in os.listdir(path):
        images.append(PIL.Image.open(os.path.join(path, img)))
    return images


if __name__ == "__main__":
    list_model = os.listdir("/content/textual_inversion/models")
    clip_eval = CLIPEvaluator(device="cuda")
    scores = {}
    img_similarities = []
    txt_similarities = []
    
    for ti_model in list_model:
        model_id = "runwayml/stable-diffusion-v1-5"
        if "black" in ti_model:
            src_images = read_dir_image("/content/textual_inversion/data/background_dark")
        elif "white" in ti_model:
            src_images = read_dir_image("/content/textual_inversion/data/background_white")
        else:
            src_images = read_dir_image("/content/textual_inversion/data/background_original")
        folder_image = f"eval_{ti_model.split('.')[0]}"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        pipe.load_textual_inversion(os.path.join("/content/textual_inversion/models", ti_model))

        generator = torch.Generator(device="cuda")

        generate_image(
            pipe,
            init_seed=42,
            num_image=64,
            prompt="a <thucpd>",
            name=folder_image,
            generator=generator,
        )
        generated_image = read_dir_image(folder_image)
        img_similarity = clip_eval.img_to_img_similarity(src_images=src_images, generated_images=generated_image)
        txt_similarity = clip_eval.txt_to_img_similarity(text="a photo of", generated_images=generated_image)
        
        scores[ti_model] = {
            "Image Similarity": img_similarity,
            "Text Similarity": txt_similarity
        }
        
        img_similarities.append(img_similarity)
        txt_similarities.append(txt_similarity)
    
    # Display the comparison table
    print("Model\tImage Similarity\tText Similarity")
    for ti_model, score in scores.items():
        print(f"{ti_model}\t{score['Image Similarity']:.4f}\t\t{score['Text Similarity']:.4f}")
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(img_similarities, txt_similarities)
    plt.title("Image Similarity vs Text Similarity")
    plt.xlabel("Image Similarity")
    plt.ylabel("Text Similarity")
    plt.show()