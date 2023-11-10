import argparse
import torch
import torch.nn as nn
import torchvision
import coremltools as ct 
import onnx
from onnx_coreml import convert
from model.net import DGCNet 


def parse():
    parser = argparse.ArgumentParser(description='Transform Toolbox')
    parser.add_argument('--model', metavar='ARCH', default='DGCNet')
    parser.add_argument('--ckpt', type=str, metavar='PATH', help='path to checkpoint')
    parser.add_argument('--onnx', action='store_true', default=False, help='export onnx')
    parser.add_argument('--coreml', action='store_true', default=False, help='export coreml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    model = DGCNet()

    # model initialized with pretrained checkpoint 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(args.ckpt, map_location=device)['state_dict'])

    
    # convert the inference mode 
    model.eval()
    model = model.to(device)

    dummy_input1 = torch.randn(1, 3, 240, 240)
    dummy_input2 = torch.randn(1, 3, 240, 240)

    if args.onnx:
        torch.onnx.export(model, (dummy_input1,dummy_input2), args.model+'.onnx', verbose=False )
        print('successfully export onnx')
    
    if args.coreml:
        example_input =  [dummy_input1, dummy_input2]
        onnx_model = onnx.load(args.model+'.onnx')
        coreml_model = convert(onnx_model, example_input)
        coreml_model.save(args.model+'.mlmodel')
        print('successfully export coreml')
    