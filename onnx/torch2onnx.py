import sys
sys.path.append('../')
import numpy as np
import torch
from cfg import Cfg
from student_train import load_deeplabv3plus_model

crop_size = 35
H = int(1216/4)-crop_size
W = int(1936/4)
teacher_weight_path = '../best2.pth'
student_weight_path = '../best2.pth'

def convert_to_onnx(config, batch_size=1, teacher=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if teacher:
        model = load_deeplabv3plus_model(config, teacher=True, teaher_weight=teacher_weight_path)
    else:
        model = load_deeplabv3plus_model(config, teacher=None, student_weight=student_weight_path)
    model.to(device)
    #model.load_state_dict(torch.load(weight_path)['model_state'])
    print('Loading weights Done!')
    
    input_layer_names = ["input"]
    output_layer_names = ["output"]
    x = torch.randn(1, 3, H, W).to(device)
    onnx_file_name = "v3plus{}_{}.onnx".format(H, W)
    
    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      verbose=True,
                      opset_version=11,
                      input_names=input_layer_names,
                      output_names=output_layer_names)
    print('Onnx model exporting done as filename {}'.format(onnx_file_name))
    
if __name__ == '__main__':
    batch_size = 1
    config = Cfg
    teacher = None
    convert_to_onnx(config, batch_size, teacher=teacher)


