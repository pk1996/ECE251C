def build_model(args):
    if args['type'] == 'mag_only':
        from .model1 import build
        return build(args)
    else:
        from .model2 import build
        return build(args)