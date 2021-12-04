import os
import torch
from network.student.modeling import deeplabv3_mobilenet
from network.teacher.modeling import deeplabv3plus_mobilenet
from solver import Solver
from cfg import Cfg

def load_deeplabv3plus_model(config, teacher=None, teaher_weight=None, student_weight=None):
    model_map = {
            #'deeplabv3_resnet50': deeplabv3_resnet50,
            #'deeplabv3plus_resnet50': deeplabv3plus_resnet50,
            #'deeplabv3_resnet101': deeplabv3_resnet101,
            #'deeplabv3plus_resnet101': deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': deeplabv3plus_mobilenet}
    if teacher:
        model = model_map['deeplabv3plus_mobilenet'](num_classes=config.num_classes, output_stride=config.output_stride)
        model.load_state_dict(torch.load(teaher_weight)['model_state'])
        print('teacher weight loading weight........')
        return model
    else:
        model = model_map['deeplabv3_mobilenet'](num_classes=config.num_classes, output_stride=config.output_stride)
        if student_weight:
            model.load_state_dict(torch.load(student_weight)['model_state'])
            print('student weight loading weight........')
        return model
    
def main():
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    num_workers = 1 # os.cpu_count()
    teacher_weight = 'best_os16.pth'
    student_weight = 'best2.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    student_model = load_deeplabv3plus_model(cfg, teacher=None, student_weight=student_weight)
    teacher_model = load_deeplabv3plus_model(cfg, teacher=True, teaher_weight=teacher_weight)
    teacher_model.to(device)
    student_model.to(device)
 
    solvers = Solver(cfg, teacher_model)
    solvers.train(config=cfg,
          smodel=student_model,
          device=device,
          num_workers=num_workers,
          pin_memory=True)
        
    
if __name__ == '__main__':
    main()


