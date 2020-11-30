def build_backbone(config, *args, **kwargs):
    if config.network.BACKBONE.name == "resnetv1b":
        from models.backbones.resnet import ResNetV1B
        backbone = ResNetV1B(*args, **kwargs)
    elif config.network.BACKBONE.name == "resnetv1":
        from models.backbones.resnet import ResNetV1
        backbone = ResNetV1(*args, **kwargs)
    elif config.network.BACKBONE.name == "mobilenetv1":
        from models.backbones.mobilenet import MobileNetV1
        backbone = MobileNetV1(*args, **kwargs)
    else:
        raise ValueError("Backbone {} is unsupported.".format(config.network.BACKBONE.name))
    return backbone
