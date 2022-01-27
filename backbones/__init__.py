from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    fp16 = kwargs.get("fp16", False)
    num_features = kwargs.get("num_features", None)
    if name == "r18":
        return iresnet18(False, **kwargs), num_features
    elif name == "r34":
        return iresnet34(False, **kwargs), num_features
    elif name == "r50":
        return iresnet50(False, **kwargs), num_features
    elif name == "r100":
        return iresnet100(False, **kwargs), num_features
    elif name == "r200":
        return iresnet200(False, **kwargs), num_features
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs), num_features
    elif name == "mbf":
        return get_mbf(fp16=fp16, num_features=num_features), num_features
    # torchvision models
    else:
        # DO NOT move these imports! They have some conflicts with iresnet models which affects their training.
        from torch import nn
        import torchvision.models as torch_models
        if name not in torch_models.__dict__:
            raise ValueError(f"Specified architecture '{name}' is not valid.")
        if fp16 or num_features is not None:
            raise ValueError("Activating fp16 mode or specifying num_features is not supported for torchvision models.")
        kwargs.pop("fp16", False)
        kwargs.pop("num_features", None)
        model = torch_models.__dict__[name](num_classes=1, **kwargs)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            out_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            out_dim = model.classifier.weight.shape[1]
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Cannot create {name} model. It does not have a Linear 'fc' or 'classifier' layer.")
        return model, out_dim
