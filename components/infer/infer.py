from diffusers import StableDiffusionPipeline
import torch

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
        image.save("{}_{}.png".format(name, seed))


if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipe.load_textual_inversion("/home/sgpu/haohq/test/xti_hoanglv_exp_thupd_caption_black/learned_embeds-steps-1600.bin")

    generator = torch.Generator(device="cuda")

    generate_image(
        pipe,
        init_seed=40,
        num_image=4,
        prompt="a <thucpd> riding a white unicorn in a cornfield.",
        name="exp_thucpd_no_use_template_original",
        generator=generator,
    )