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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="a <thucpd> riding a unicorn in a corn field.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--name_exp", type=str, default="exp")
    
    args = parser.parse_args()
    
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipe.load_textual_inversion(args.checkpoint)

    generator = torch.Generator(device="cuda")

    generate_image(
        pipe,
        init_seed=40,
        num_image=4,
        prompt=args.prompt,
        name=args.name_exp,
        generator=generator,
    )