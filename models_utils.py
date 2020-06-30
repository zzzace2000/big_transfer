# import torch
# from arch.Inpainting.Baseline import RandomColorWithNoiseInpainter
# from arch.Inpainting.AE_Inpainting import VAEWithVarInpaintModelMean as VAEInpainter
# from arch.Inpainting.CAInpainter import CAInpainter
#
#
# def get_inpaint_model(gen_model_name, batch_size):
#     inpainter_names = [a.__name__ for a in [
#         BlurryInpainter, LocalMeanInpainter, MeanInpainter, \
#         RandomColorWithNoiseInpainter,
#     ]]
#
#     if gen_model_name in inpainter_names:
#         inpaint_model_obj = eval(gen_model_name)
#         inpaint_model = inpaint_model_obj()
#     elif gen_model_name == 'CAInpainter':
#         # have not implemented
#         inpaint_model = CAInpainter(
#             batch_size, checkpoint_dir='./inpainting_models/release_imagenet_256/')
#     elif gen_model_name == 'VAEInpainter':
#         gen_model_path = 'inpainting_models/0928-VAE-Var-hole_lr_0.0002_epochs_7'
#         inpaint_model = VAEInpainter()
#
#         inpaint_model.load_state_dict(
#             torch.load(gen_model_path, map_location=lambda storage, loc: storage),
#             strict=False)
#
#     inpaint_model.eval()
#     return inpaint_model
