#!bin/bash

python components/infer/infer.py \
        --name_exp "exp_thucpd_default" \
        --prompt "a photo of <nhim> riding a unicorn in a cornfield" \
        --checkpoint "exp_thucpd_default/learned_embeds.bin"