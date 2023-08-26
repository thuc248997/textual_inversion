from diffusers import StableDiffusionPipeline
import torch

def generate_images_with_seeds(text_to_image_model, initial_seed, num_images, text_prompt, num_inference_steps=50,
                               guidance_scale=7.5, generator_model=None, experiment_name="experiment"):
    """
    Generate a series of images based on text prompts using a text-to-image model.

    Args:
        text_to_image_model: The text-to-image generation model.
        initial_seed (int): Initial seed value for random number generation.
        num_images (int): Number of images to generate.
        text_prompt (str): Text prompt for generating images.
        num_inference_steps (int): Number of inference steps for generation.
        guidance_scale (float): Guidance scale for controlling the generation process.
        generator_model: The generator model to use, if applicable.
        experiment_name (str): Name for the experiment.

    Returns:
        None
    """
    for i in range(num_images):
        seed = initial_seed + i
        torch.manual_seed(seed)
        generator_model.manual_seed(seed)
        generated_image = generate_single_image(text_to_image_model, text_prompt, num_inference_steps,
                                                guidance_scale, generator_model)
        generated_image.save("{}_{}.png".format(experiment_name, seed))

def generate_single_image(text_to_image_model, text_prompt, num_inference_steps=50,
                          guidance_scale=7.5, generator_model=None):
    """
    Generate a single image based on a text prompt using a text-to-image model.

    Args:
        text_to_image_model: The text-to-image generation model.
        text_prompt (str): Text prompt for generating the image.
        num_inference_steps (int): Number of inference steps for generation.
        guidance_scale (float): Guidance scale for controlling the generation process.
        generator_model: The generator model to use, if applicable.

    Returns:
        PIL.Image.Image: The generated image.
    """
    return text_to_image_model(
        text_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator_model,
    ).images[0]

if __name__ == "__main__":
    # Khởi tạo model và pipe
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipe.load_textual_inversion("exp_thucpd/learned_embeds.bin")
    
    # Khởi tạo generator
    generator = torch.Generator(device="cuda")

    # Tạo hình ảnh dựa trên prompt cho từng seed
    generate_images_with_seeds(
        pipe,
        init_seed=42,
        num_image=5,
        prompt="a photo of <nhim> ridding a unicorn",
        name="exp_thucpd",
        generator=generator,
    )