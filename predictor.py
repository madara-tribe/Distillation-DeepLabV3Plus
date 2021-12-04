from tqdm import tqdm
import os
import time
import random
import numpy as np
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from data_utils.stream_metrics import StreamSegMetrics, AverageMeter
from visualizer import Visualizer
from solver import get_loader
from cfg import Cfg
from student_train import load_deeplabv3plus_model


def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    save_val_results = True
    dirname = 'preds'
    if save_val_results:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (image * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    #plt.imshow(pred),plt.show()
                    pred = np.hstack([image, target, pred])
                    #Image.fromarray(image).save('results/%d_image.png' % img_id)
                    #Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('preds/%d_pred.png' % img_id)
            
                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('preds/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples





def predict(config, device, model):
    # config
    enable_vis = False
    vis_env = 'main'
    vis = Visualizer(port=config.vis_port, env=vis_env) if enable_vis else None


    val_batch_size = 1
    num_classes = config.num_classes
    
    print('random seed')
    random_seed = config.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print('loading data....')
    _, val_loader, _ = get_loader(config, num_worker=1, pin_memory=None)
    
    # Set up model
    metrics = StreamSegMetrics(num_classes)
    
    print('setting up model....')
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                    np.int32) if enable_vis else None  # sample idxs for visualization
    model.eval()
    print('saving predict images to results floder....')
    start_time = time.time()
    val_score, ret_samples = validate(
        model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    print(metrics.to_str(val_score))
    print("Inference Latency (ms) until saved is", (time.time() - start_time)*1000, "[ms]")

if __name__ == '__main__':
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    teacher_weight_path = 'best_os16.pth'
    student_weight_path = 'best2.pth'
    teacher=False
    if teacher:
        model = load_deeplabv3plus_model(cfg, teacher=True, teaher_weight=teacher_weight_path)
    else:
        model = load_deeplabv3plus_model(cfg, teacher=None, student_weight=student_weight_path)
    model.to(device)
    predict(config=cfg, 
         device=device, 
         model=model)


