

def freeze_modules(model, modules_2_freeze):
    for name, child in model.named_children():
        if name in modules_2_freeze:
            child.requires_grad_(False)
        else:
            freeze_modules(child, modules_2_freeze)