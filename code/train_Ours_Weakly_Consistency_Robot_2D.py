import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from config import get_config
import os,sys
os.chdir(sys.path[0])
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/robotic', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='robotic/Interpolation_Consistency_Training_8', help='experiment_name')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='ViT_Seg', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

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

parser.add_argument('--ict_alpha', type=int, default=0.5,
                    help='ict_alpha')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model.load_from(config)

    ema_model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    ema_model.load_from(config)
    
    for param in ema_model.parameters():
        param.detach_()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    print("Total silices is: {}".format(total_slices))

    trainloader = DataLoader(db_train, batch_size=batch_size,shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    pce_loss = CrossEntropyLoss(ignore_index=2)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # condition = label_batch[:] == 255
            # replacement = torch.tensor(2, dtype=torch.uint8, device='cuda:0')
            # label_batch[:] = torch.where(condition, replacement, label_batch[:])
            # ICT mix factors
            ict_mix_factors = np.random.beta(
                args.ict_alpha, args.ict_alpha, size=(batch_size//2, 1, 1, 1))
            ict_mix_factors = torch.tensor(
                ict_mix_factors, dtype=torch.float).cuda()
            volume_batch_0 = volume_batch[0:batch_size//2, ...]
            label_batch_0 = label_batch[0:batch_size//2, ...]
            volume_batch_1 = volume_batch[batch_size//2:, ...]
            label_batch_1 = label_batch[batch_size//2:, ...]
            batch_label_mixed = torch.zeros_like(label_batch_0)
            # Mix images
            batch_ux_mixed = volume_batch_0 * \
                (1.0 - ict_mix_factors) + \
                volume_batch_1 * ict_mix_factors
            # for i in range(batch_label_mixed.shape[0]):
            #     if ict_mix_factors[i] < 0.5 :
            #         batch_label_mixed[i] = label_batch_0[i]
            #     else :
            #         batch_label_mixed[i] = label_batch_1[i]
            input_volume_batch = torch.cat(
                [volume_batch, batch_ux_mixed], dim=0)
            outputs = model(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(volume_batch_0), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(volume_batch_1), dim=1)
                batch_pred_mixed = ema_output_ux0 * \
                    (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors
            pseudo_label = torch.argmax(batch_pred_mixed, dim=1, keepdim=False)
            loss_ce_1 = pce_loss(outputs[:volume_batch.shape[0]],
                              label_batch[:volume_batch.shape[0]][:].long())
            # loss_ce_2 = pce_loss(outputs[volume_batch.shape[0]:],
            #                      batch_label_mixed.long())
            loss_ce_3 = ce_loss(outputs[volume_batch.shape[0]:],
                              pseudo_label.long())
            loss_dice = dice_loss(
                outputs_soft[volume_batch.shape[0]:], pseudo_label.unsqueeze(1))
            consistency_loss = torch.mean(
                (outputs_soft[volume_batch.shape[0]:] - batch_pred_mixed)**2)
            #pseudo_loss = loss_ce_2 + loss_ce_3 + loss_dice + consistency_loss
            pseudo_loss = loss_ce_3 +loss_dice + consistency_loss
            consistency_weight = get_current_consistency_weight(iter_num//len(trainloader))
            loss = loss_ce_1 + consistency_weight * pseudo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              pseudo_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f c_w : %f'
                %(iter_num, loss.item(),consistency_weight))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                image = batch_ux_mixed[1, 0:1, :, :]
                writer.add_image('train/Mixed_Unlabeled',
                                 image, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}".format(
        args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
