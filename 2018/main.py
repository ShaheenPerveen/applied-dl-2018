from __future__ import division

import os, sys, pdb, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, accuracy2
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets, models
import random
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets, models
import random
import sys

import kmodels
from kmodels import *
from ktransforms import *
from kdataset import *


# model_names = sorted(name for name in kmodels.__dict__
#   if name.islower() and not name.startswith("__")
#   and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--data_path', type=str, help='Path to dataset')

# parser.add_argument('--num_classes', type=int, default=12, help='Number of Classes in data set.')
# parser.add_argument('--data_path', default='d:/db/data/seedlings/train/', type=str, help='Path to train dataset')
# parser.add_argument('--test_data_path', default='d:/db/data/seedlings/test/', type=str, help='Path to train dataset')
# parser.add_argument('--dataset', type=str, default='seedlings', choices=['seedlings'], help='Choose between Cifar10/100 and ImageNet.')



parser.add_argument('--num_classes', type=int, default=2, help='Number of Classes in data set.')
parser.add_argument('--data_path', default='d:/db/data/IDC_regular_ps50_idx5/', type=str, help='Path to train dataset')
parser.add_argument('--dataset', type=str, default='IDC_regular_ps50_idx5', choices=['IDC_regular_ps50_idx5'], help='Choose between data sets')


# parser.add_argument('--num_classes', type=int, default=2, help='Number of Classes in data set.')
# parser.add_argument('--data_path', default='d:/db/data/ISIC2017/train/', type=str, help='Path to train dataset')
# parser.add_argument('--dataset', type=str, default='ISIC2017', choices=['ISIC2017'], help='Choose between data sets')

# parser.add_argument('--num_classes', type=int, default=2, help='Number of Classes in data set.')
# parser.add_argument('--data_path', default='d:/db/data/bone/TCB_Challenge_Data/train/', type=str, help='Path to train dataset')
# parser.add_argument('--dataset', type=str, default='bone', choices=['bone'], help='Choose between data sets')


# parser.add_argument('--arch', metavar='ARCH', default='resnext', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
# parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
# parser.add_argument('--learning_rate', type=float, default=0.00005 * 2 * 2, help='The Learning Rate.')
parser.add_argument('--learning_rate', type=float, default= 0.0005 * 2 * 2, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./logs/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed', default=2222)
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true', default=True)

parser.add_argument('--validationRatio', type=float, default=0.85, help='test Validation Split.')
# parser.add_argument('--data_path_test', default='/home/data/cat-dog/test/', type=str, help='Path to test dataset')
parser.add_argument('--imgDim', default=3, type=int, help='number of Image input dimensions')
parser.add_argument('--img_scale', default=224, type=int, help='Image scaling dimensions')


args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
  args.manualSeed = 999

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main():
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  print('Dataset: {}'.format(args.dataset.upper()))

  if args.dataset=="seedlings" or args.dataset=="bone":
    classes, class_to_idx, num_to_class, df = GenericDataset.find_classes(args.data_path)
  if args.dataset=="ISIC2017":
    classes, class_to_idx, num_to_class, df = GenericDataset.find_classes_melanoma(args.data_path)

  if args.dataset=="IDC_regular_ps50_idx5":
    classes, class_to_idx, num_to_class, df = GenericDataset.find_classes_breast(args.data_path)

  df.head(3)

  args.num_classes=len(classes)
  # Init model, criterion, and optimizer
  # net = models.__dict__[args.arch](num_classes)
  net= kmodels.simpleXX_generic(num_classes=args.num_classes, imgDim=args.imgDim)
  # net= kmodels.vggnetXX_generic(num_classes=args.num_classes,  imgDim=args.imgDim)
  # net= kmodels.vggnetXX_generic(num_classes=args.num_classes,  imgDim=args.imgDim)
  # net= kmodels.dpn92(num_classes=args.num_classes)
  # net= kmodels.inception_v3(num_classes=args.num_classes)
  # print_log("=> network :\n {}".format(net), log)

  real_model_name = (type(net).__name__)
  print("=> Creating model '{}'".format(real_model_name))
  # if real_model_name is "Inception3":
  #   net = inception_v3(pretrained=True)
  #   net.fc = nn.Linear(2048, args.num_classes)

  import datetime

  exp_name = datetime.datetime.now().strftime(real_model_name + '_' + args.dataset + '_%Y-%m-%d_%H-%M-%S')
  print('Training ' + real_model_name + ' on {} dataset:'.format(args.dataset.upper()))

  mPath = args.save_path + '/' + args.dataset + '/' + real_model_name + '/'
  args.save_path_model = mPath
  if not os.path.isdir(args.save_path_model):
      os.makedirs(args.save_path_model)

  log = open(os.path.join(mPath, 'seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print("Random Seed: {}".format(args.manualSeed))
  print("python version : {}".format(sys.version.replace('\n', ' ')))
  print("torch  version : {}".format(torch.__version__))
  print("cudnn  version : {}".format(torch.backends.cudnn.version()))

  # Init dataset
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)
  normalize_img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])

  train_trans = transforms.Compose([
      transforms.RandomSizedCrop(args.img_scale),
      PowerPIL(),
      transforms.ToTensor(),
      # normalize_img,
      RandomErasing()
  ])

  ## Normalization only for validation and test
  valid_trans = transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(args.img_scale),
      transforms.ToTensor(),
      # normalize_img
  ])

  test_trans=valid_trans

  if args.dataset == "IDC_regular_ps50_idx5":
    full_data = df.sample(frac=0.20)  # save for testing (not validation)
    # full_data = full_data.sample(frac=0.10) # DATA SET IS VERY LARGE use a sample
    test_data = df[~df['file'].isin(full_data['file'])]
    test_set = GenericDataset(test_data, args.data_path, transform=valid_trans)

    # Train validation split
    train_data = full_data.sample(frac=args.validationRatio)
    valid_data = full_data[~full_data['file'].isin(train_data['file'])]
    train_set = GenericDataset(train_data, args.data_path, transform=train_trans)
    valid_set = GenericDataset(valid_data, args.data_path, transform=valid_trans)

  else:
    train_data = df.sample(frac=args.validationRatio)
    valid_data = df[~df['file'].isin(train_data['file'])]

    train_set = GenericDataset(train_data, args.data_path, transform=train_trans)
    valid_set = GenericDataset(valid_data, args.data_path, transform=valid_trans)

  t_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
  v_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
  # test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

  dataset_sizes = {
      'train': len(t_loader.dataset),
      'valid': len(v_loader.dataset)
  }
  print(dataset_sizes)
  # net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
  criterion = torch.nn.CrossEntropyLoss()

  # optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
  #               weight_decay=state['decay'], nesterov=True)

  optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
  # optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
  #                             weight_decay=state['decay'], nesterov=True)
  # optimizer = torch.optim.Adam(net.parameters(), lr=state['learning_rate'])

  if args.use_cuda:
    net.cuda()
    criterion.cuda()

  recorder = RecorderMeter(args.epochs)
  # optionally resume from a checkpoint
  if args.evaluate:
    validate(v_loader, net, criterion, log)
    return
  if args.tensorboard:
      configure("./logs/runs/%s" % (exp_name))

  print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

  # Main loop
  start_training_time = time.time()
  training_time=time.time()
  start_time = time.time()
  epoch_time = AverageMeter()
  for epoch in tqdm(range(args.start_epoch, args.epochs)):
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
    # print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

    tqdm.write('\n==>>Epoch=[{:03d}/{:03d}]], {:s}, LR=[{}], Batch=[{}]'.format(epoch, args.epochs, time_string(),
                                                                                state['learning_rate'],
                                                                                args.batch_size) + ' [Model={}]'.format((type(net).__name__), ), log)

    # train for one epoch
    train_acc, train_los = train(t_loader, net, criterion, optimizer, epoch, log)
    val_acc,   val_los   = validate(v_loader, net, criterion, epoch, log)
    is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    training_time=time.time() - start_training_time
    recorder.plot_curve(os.path.join(mPath, real_model_name + '_' + exp_name + '.png'),training_time, net, real_model_name,dataset_sizes,
                        args.batch_size, args.learning_rate,args.dataset,args.manualSeed,args.num_classes)

    if float(val_acc) > float(95.0):
      print("*** EARLY STOP ***")
      df_pred=testSeedlingsModel(args.test_data_path, net, num_to_class, test_trans)
      model_save_path= os.path.join(mPath, real_model_name + '_' + str(val_acc) + '_' + str(val_los) + "_" + str(epoch))

      df_pred.to_csv(model_save_path + "_sub.csv", columns=('file', 'species'), index=None)
      torch.save(net.state_dict(),model_save_path + '_.pth')

      save_checkpoint({
          'epoch': epoch + 1,
          # 'arch': args.arch,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer': optimizer.state_dict(),
        }, is_best, mPath ,  str(val_acc) + '_' + str(val_los) + "_" + str(epoch) + '_checkpoint.pth.tar')

  log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)

    # a fix for inception v3, RuntimeError: log_softmax(): argument 'input' (position 1) must be Variable, not tuple
    # if isinstance(output, tuple):
    #   loss = sum((criterion(o, target_var) for o in output))
    # else:
    #   loss = criterion(output, target_var)

    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy2(output.data, target, topk=(1, 1))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
            'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
            'Loss {loss.val:.4f} ({loss.avg:.4f})   '
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
  print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
  # log to TensorBoard
  if args.tensorboard:
      log_value('train_loss', losses.avg, epoch)
      log_value('train_error', top1.avg, epoch)
  return top1.avg, losses.avg

def validate(val_loader, model, criterion,epoch, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy2(output.data, target, topk=(1, 1))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

  print_log('  **VAL** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  if args.tensorboard:
      log_value('val_loss', losses.avg, epoch)
      log_value('val_acc', top1.avg, epoch)
  return top1.avg, losses.avg


def testSeelingsImageLoader(image_name, test_trans):
  """load image, returns cuda tensor"""
  #     image = Image.open(image_name)
  image = Image.open(image_name).convert('RGB')
  image = test_trans(image)
  #     image = Variable(image, requires_grad=True)
  image = image.unsqueeze(0)
  if args.use_cuda:
    #         print ("cuda")
    image.cuda()
  return image


def testSeedlingsModel(test_dir, local_model, num_to_class, test_trans):
  sample_submission = pd.read_csv('sample_submission.csv')
  sample_submission.columns = ['file', 'species']

  if args.use_cuda:
    local_model.cuda()
  local_model.eval()

  columns = ['file', 'species']
  df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)

  for index, row in (sample_submission.iterrows()):
    #         for file in os.listdir(test_dir):
    currImage = os.path.join(test_dir, row['file'])
    if os.path.isfile(currImage):
      X_tensor_test = testSeelingsImageLoader(currImage, test_trans=test_trans)
      #             print (type(X_tensor_test))
      if args.use_cuda:
        X_tensor_test = Variable(X_tensor_test.cuda())
      else:
        X_tensor_test = Variable(X_tensor_test)

        # get the index of the max log-probability
      predicted_val = (local_model(X_tensor_test)).data.max(1)[1]  # get the index of the max log-probability
      #             predicted_val = predicted_val.data.max(1, keepdim=True)[1]
      p_test = (predicted_val.cpu().numpy().item())
      df_pred = df_pred.append({'file': row['file'], 'species': num_to_class[int(p_test)]}, ignore_index=True)

  return df_pred

def extract_features(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output, features = model([input_var])

    pdb.set_trace()

    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy2(output.data, target, topk=(1, 5))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

  print_log('  **VAL** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)


  return top1.avg, losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


if __name__ == '__main__':
  main()
