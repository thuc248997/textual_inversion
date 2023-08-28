import os
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from torchvision import transforms
import numpy as np
import clip
from diffusers import StableDiffusionPipeline

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
        """
        Initialize the CLIPEvaluator.

        Args:
            device (str): Device to run the evaluation on (e.g., "cuda" or "cpu").
            clip_model (str): Name of the CLIP model to load.
        """
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = self.clip_preprocess
        self.generator_preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
            + self.clip_preprocess.transforms[:2]
            + self.clip_preprocess.transforms[4:]
        )

    def tokenize_text(self, strings: list) -> torch.Tensor:
        """
        Tokenize a list of text strings.

        Args:
            strings (list): List of text strings to tokenize.

        Returns:
            torch.Tensor: Tokenized text tensor.
        """
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        """
        Encode tokenized text.

        Args:
            tokens (list): Tokenized text tensor.

        Returns:
            torch.Tensor: Encoded text features.
        """
        return self.clip_model.encode_text(tokens)

    @torch.no_grad()
    def get_image_features(self, images) -> torch.Tensor:
        """
        Get image features.

        Args:
            images: List of PIL Images or tensor of [-1, 1] images.

        Returns:
            torch.Tensor: Image features.
        """
        if isinstance(images[0], Image.Image):
            images = torch.stack([self.clip_preprocess(image) for image in images]).to(
                self.device
            )
        else:
            images = self.generator_preprocess(images).to(self.device)
        return self.clip_model.encode_image(images)

    def get_text_features(self, text: str) -> torch.Tensor:
        """
        Get text features.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: Text features.
        """
        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens)
        return text_features

    def image_to_image_similarity(self, src_images, generated_images) -> float:
        """
        Calculate image-to-image similarity.

        Args:
            src_images: Source images.
            generated_images: Generated images.

        Returns:
            float: Image-to-image similarity score.
        """
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

    def text_to_image_similarity(self, text, generated_images, reduction=True) -> float:
        """
        Calculate text-to-image similarity.

        Args:
            text (str): Input text.
            generated_images: Generated images.
            reduction (bool): Whether to reduce similarity scores to a single value.

        Returns:
            float or np.ndarray: Text-to-image similarity score(s).
        """
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        if reduction:
            return cosine_similarity(text_features, gen_img_features).mean().item()
        else:
            return cosine_similarity(text_features, gen_img_features)

def read_dir_image(path: str) -> list:
    """
    Read images from a directory.

    Args:
        path (str): Path to the directory containing images.

    Returns:
        list: List of PIL Image objects.
    """
    images = []
    for img in os.listdir(path):
        images.append(Image.open(os.path.join(path, img)))
    return images


if __name__ == "__main__":
    list_model = os.listdir("/content/sd-tools/textual_inversion_training")
    clip_eval = CLIPEvaluator(device="cuda")
    scores = {}
    for ti_model in list_model:
        model_id = "runwayml/stable-diffusion-v1-5"
        if "fixed_bg" in ti_model:
            src_images = read_dir_image("/content/sd-tools/datas/keep_bg")
        else:
            src_images = read_dir_image("/content/sd-tools/datas/origin")
        folder_image = f"eval_{ti_model.split('.')[0]}"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        pipe.load_textual_inversion(os.path.join("/content/sd-tools/textual_inversion_training", ti_model))

        generator = torch.Generator(device="cuda")

        generate_image(
            pipe,
            init_seed=42,
            num_image=64,
            prompt="a photo of <thupc>",
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