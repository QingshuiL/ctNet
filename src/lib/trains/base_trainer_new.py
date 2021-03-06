from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from apex import amp
#from apex.parallel import DistributedDataParallel as DDP
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch[0]['input'])   ####
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model = model
    # self.model self.loss

  def set_device(self, gpus, chunk_sizes, device): 
    assert len(gpus) == 1, 'The number of gpus exceeds 1'
    self.model = self.model.to(device)
    self.loss = self.loss.to(device)

    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)
    
    self.model, self.optimizer = amp.initialize(self.model, \
                self.optimizer, opt_level='O1')


  def run_epoch(self, phase, epoch, data_loader):
    model = self.model
    criterion = self.loss
    if phase == 'train':
      model.train()
    else:
      if len(self.opt.gpus) > 1:
        model = model.module
      model.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      for i in range(opt.num_stacks):
        for k in batch[i]:
          if k != 'meta':
            batch[i][k] = batch[i][k].to(device=opt.device, non_blocking=True)    
      # output, loss, loss_stats = model(batch)
      outputs = model(batch[0]['input'])
      loss, loss_stats = criterion(outputs, batch)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          #loss_stats[l].mean().item(), batch['input'].size(0))
          loss_stats[l].mean().item(), 3)
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, outputs[1][-1], iter_id)    ## output
      
      if opt.test:
        self.save_result(outputs[1][-1], batch, results)  ## output
      del outputs, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError

  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)