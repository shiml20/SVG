## Training Scripts of SVG-Autoencoder

### Installation

[Taming-Transformers](https://github.com/CompVis/taming-transformers?tab=readme-ov-file) is needed for training. 
    
    Get it by running:
    ```
    git clone https://github.com/CompVis/taming-transformers.git
    cd taming-transformers
    pip install -e .
    ```

    Then modify ``./taming-transformers/taming/data/utils.py`` to meet torch 2.x:
    ```
    export FILE_PATH=./taming-transformers/taming/data/utils.py
    sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' "$FILE_PATH"
    ```


### Train

1. Modify training config as you need.

2. Run training by:

    ```
    bash run_train.sh svg/configs/example_svg_autoencoder_vitsp.yaml
    ```
    Your training logs and checkpoints will be saved in the `logs` folder. We train SVG-Autoencoder with 8 H800 GPUs.


### Acknowledgement

SVG's training is mainly built upon [VA-VAE](https://github.com/hustvl/LightningDiT/) and [LDM](https://github.com/CompVis/latent-diffusion/tree/main). Thanks for the great work!
