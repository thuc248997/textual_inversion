#!bin/bash

python components/infer/infer.py \
        --name_exp "exp_nhim_no_use_template" \
        --prompt <nhim> riding a white unicorn in cornfield \
        --checkpoint "exp_nhim_no_use_template/learned_embeds-steps-2000.bin"

python components/infer/infer.py \
        --model_id "runwayml/stable-diffusion-v1-5" \
        --prompt '<thucpd> riding a white unicorn in cornfield' \
        --checkpoint "exp_thucpd_with_caption_rm_background/learned_embeds.bin"\
        --name_exp "
python components/infer/infer.py \
        --name_exp "exp_thucpd_default" \
        --prompt 'a <thucpd> on a white unicorn in corn field' \
        --checkpoint "exp_thucpd_default/learned_embeds.bin"