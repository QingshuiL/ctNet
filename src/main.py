from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os



import torch
import torch.utils.data
from torchvision.transforms import Compose
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.ctdet import CtdetTrainer

from datasets.transforms.transforms import MultiScale, RandomCrop, ColorJitter, ToTensor

Train_mean = (0.485, 0.456, 0.406)
Train_std = (0.229, 0.224, 0.225)
Train_crop_size = (1024, 1024)
Train_transforms = Compose([
    MultiScale(scale=(1, 1.15, 1.25, 1.35, 1.5)),
    ColorJitter(),
    ToTensor(),
    RandomCrop(Train_crop_size),
    # MaskIgnore(Config.Train.mean),
    # FillDuck(),
    # HorizontalFlip(),
    # Normalize(Config.Train.mean, Config.Train.std),
    # ToHeatmap(scale_factor=Config.Train.scale_factor)
])


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)                            # 数据类实例化
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)            # 参数类实例
  print(opt)

  logger = Logger(opt)                                                    # 日志类实例

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  # 得到网络结构
  model = create_model(opt.arch, opt.heads, opt.head_conv)                # 网络结构实例
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)                # 优化器 Adam
  start_epoch = 0
  if opt.load_model_stage1 != '':                                                # 用于加载已经训练好的模型 或 继续训练模型
    model, optimizer, start_epoch = load_model(
      model, opt.load_model_stage1, optimizer, opt.resume, opt.lr, opt.lr_step)

  
  trainer = CtdetTrainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val', Train_transforms), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train', Train_transforms), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader) ############ 开始传入图片索引
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
