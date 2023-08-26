import os
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from torchvision import transforms
import numpy as np
import clip


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
    clip_eval = CLIPEvaluator(device="cuda")
    src_images = read_dir_image("data")
    generate_images = read_dir_image("generate")
    print(
        clip_eval.image_to_image_similarity(
            src_images=src_images, generated_images=generate_images
        )
    )
    text = "a photo of"
    print(clip_eval.text_to_image_similarity(text=text, generated_images=generate_images))