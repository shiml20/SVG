import torchvision.transforms as T

TRANSFORMS = dict()


def register_transform(transform):
    name = transform.__name__
    if name in TRANSFORMS:
        raise RuntimeError(f'Transform {name} has already registered.')
    TRANSFORMS.update({name: transform})


def get_transform(type, resolution):
    transform = TRANSFORMS[type](resolution)
    transform = T.Compose(transform)
    transform.image_size = resolution
    return transform


@register_transform
def default_train(n_px):
    transform = [
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(n_px),  # Image.BICUBIC
        T.CenterCrop(n_px),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ]
    return transform


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@register_transform
def dit_train(n_px):
    transform = [
        T.Lambda(lambda img: center_crop_arr(img, n_px)),
        # T.Resize(n_px),  # Image.BICUBIC
        # T.CenterCrop(n_px),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5], inplace=True),
    ]
    return transform