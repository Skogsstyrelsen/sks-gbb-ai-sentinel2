import torch
import utils.model.AttentionPixelClassifier as attentionPixelClassifier

def load_model(opt,train=False):
    if opt.device != 'cpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")


    if str(opt.algorithm).lower() == 'attentioncenet':
        model = attentionPixelClassifier.AttentionCENet(
        input_numChannels=opt.input_channels[0],
        output_numChannels=opt.output_channels).to(device)

    print(f"About to load model {opt.weights}, input_ch = {opt.input_channels[0]}, output_ch = {opt.output_channels}")
    if opt.weights != '':
        model.load_state_dict(torch.load(opt.weights,map_location=device,weights_only=False))
    if not train:
        model.eval()
    return model, device