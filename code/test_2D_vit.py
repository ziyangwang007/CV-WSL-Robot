import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from skimage.color import rgb2gray
# from networks.efficientunet import UNet
from networks.net_factory import net_factory


from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')


parser.add_argument('--root_path', type=str,
                    default='../data/robotic', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='robot/CRF_ViT', help='experiment_name')
# parser.add_argument('--model', type=str,
#                     default='vnet', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='fold')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
# parser.add_argument('--sup_type', type=str, default="label",help='label')
parser.add_argument('--sup_type', type=str, default="scribble",help='label/scribble')


parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/test/{}".format(case), 'r')
    image = h5f['image'][:]
    image = rgb2gray(image)/255.0
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (224 / x, 224 / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 224, y / 224), order=0)
        prediction = pred
    # case = case.replace(".h5", "")
    # org_img_path = "../data/ACDC_training/{}.nii.gz".format(case)
    # org_img_itk = sitk.ReadImage(org_img_path)
    # spacing = org_img_itk.GetSpacing()

    # first_metric = calculate_metric_percase(
    #     prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    # second_metric = calculate_metric_percase(
    #     prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    # third_metric = calculate_metric_percase(
    #     prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    # return first_metric, second_metric, third_metric


def Inference(FLAGS):
    all_volumes = os.listdir(FLAGS.root_path + "/test")
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}/{}/ViT_predictions/".format(
        FLAGS.exp,FLAGS.sup_type)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    
    net = ViT_seg(config, img_size=[224, 224],
                     num_classes=args.num_classes).cuda()
    net.load_from(config)



    save_mode_path = os.path.join(
        snapshot_path, 'ViT_best_model.pth')
    save_mode_path = r'D:\robot\iccv-surgical\model\robotic\Interpolation_Consistency_Training_4\ViT_Seg\ViT_Seg_best_model.pth'
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    image_list = []
    for file in all_volumes:
        if file.endswith(".h5"):
            image_list.append(file)

    # first_total = 0.0
    # second_total = 0.0
    # third_total = 0.0
    for case in tqdm(image_list):
        print(case)
        test_single_volume(case, net, test_save_path, FLAGS)
    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
    # print(avg_metric)
    # print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)
    # return ((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)[0]


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    mean_dice = Inference(FLAGS)