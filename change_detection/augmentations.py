import albumentations as albu
import albumentations.pytorch as apt

DS_3DCD_MIN_VALUE = -25
DS_3DCD_MAX_VALUE = 30
DS_3DCD_MEAN = [0.5896145210542503, 0.6210658017517566, 0.591661801751776]
DS_3DCD_STD = [0.1898555514094201, 0.19114699478664082, 0.21242997453209553]

def get_training_augmentations(m = [0,0,0], s = [1,1,1]):
    transform = [
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.GaussNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),

        albu.Normalize(mean = m, std = s),
        apt.ToTensorV2(),
    ]
    return albu.Compose(transform, additional_targets={'t2': 'image', 'mask3d': 'mask'})

def get_basic_augmentations(m = [0,0,0], s = [1,1,1]):
    transforms = [
        albu.Normalize(mean = m, std = s),
        apt.ToTensorV2(),
    ]

    return albu.Compose(transforms, additional_targets={'t2': 'image', 'mask3d': 'mask'})
