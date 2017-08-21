"""Training and validation code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Data
import Batch
import Utils

import matplotlib.pyplot as plt

from nets.SqueezeNet import SqueezeNet
from nets.SqueezeNetOrig import SqueezeNetOrig
import torch
import torch.nn.utils as nnutils
from torch.autograd import Variable

class MTL:
    def __init__(self, load_path=None):
        self.net1 = SqueezeNet().cuda()
        self.criterion1 = torch.nn.MSELoss().cuda()
        self.optimizer1 = torch.optim.Adadelta(self.net1.parameters())
        self.loss = Utils.LossLog()

        if load_path is not None:
            save_data = torch.load(load_path)
            self.net1.load_state_dict(save_data)

    def forward(self, camera_data, metadata, target_data):
        self.optimizer1.zero_grad()
        outputs1 = self.net1(Variable(camera_data), Variable(metadata)).cuda()
        self.loss1 = self.criterion1(outputs1, Variable(target_data))
        self.loss.add(self.loss1.data[0])
    
    def backward(self):
        self.loss1.backward()
        nnutils.clip_grad_norm(self.net1.parameters(), 1.0)
        self.optimizer1.step()
        
    def get_average():
        return self.loss.average()

    def reset_loss():
        self.loss = Utils.LossLog()

    def save_net(save_name, epoch):
        Utils.save_net(save_name + epoch, self.net1)

class NonMTL:
    def __init__(self, load_path=None):
        self.net1 = SqueezeNetOrig().cuda()
        self.criterion1 = torch.nn.MSELoss().cuda()
        self.optimizer1 = torch.optim.Adadelta(self.net1.parameters())
        self.loss = Utils.LossLog()
        self.net1.train()  # Train mode

        if load_path is not None:
            save_data = torch.load(load_path)
            self.net1.load_state_dict(save_data)

    def forward(self, camera_data, target_data):
        self.optimizer1.zero_grad()
        outputs1 = self.net1(Variable(camera_data)).cuda()
        self.loss1 = self.criterion1(outputs1, Variable(target_data))
        self.loss.add(self.loss1.data[0])
    
    def backward(self):
        self.loss1.backward()
        nnutils.clip_grad_norm(self.net1.parameters(), 1.0)
        self.optimizer1.step()
        
    def get_average():
        return loss.average()

    def reset_loss():
        self.loss = Utils.LossLog()

def main():
    logging.basicConfig(filename='training.log', level=logging.DEBUG)
    logging.debug(ARGS)  # Log arguments

    Utils.csvwrite('VALIDATION_LOSS.csv', ['mtl', 
                   'direct', 'follow',
                   'furtive', 'control'])
    epoch = ARGS.epoch
    if epoch == 0:
        mtl = MTL()
        direct = NonMTL()
        follow = NonMTL()
        furtive = NonMTL()
        control = NonMTL()
    else:
        epoch = str(epoch)
        mtl = MTL(load_path='save/mtl'+epoch)
        direct = NonMTL(load_path='save/mtl'+epoch)
        follow = NonMTL(load_path='save/follow'+epoch)
        furtive = NonMTL(load_path='save/furtive'+epoch)
        control = NonMTL(load_path='save/control'+epoch)
        epoch = int(epoch)

    # Set Up PyTorch Environment
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(ARGS.gpu)
    torch.cuda.device(ARGS.gpu)

    data = Data.Data()
    batch = Batch.Batch()

    rate_counter = Utils.RateCounter()

    try:
        logging.debug('Starting training epoch #{}'.format(epoch))

        print_counter = Utils.MomentCounter(ARGS.print_moments)

        while not data.train_index.epoch_complete:  # Epoch of training
            # Extract all data
            camera_data, metadata, target_data = batch.fill(data, data.train_index)
            dcamera, focamera, fucamera = camera_data.chunk(3, 0)
            dtarget, fotarget, futarget = target_data.chunk(3, 0)

            mtl.forward(camera_data, metadata, target_data)
            mtl.backward()

            control.forward(camera_data, target_data)
            control.backward()

            direct.forward(dcamera, dtarget)
            direct.backward()

            follow.forward(focamera, fotarget)
            follow.backward()

            furtive.forward(fucamera, futarget)
            furtive.backward()

            if print_counter.step(data.train_index):
                print('mode = train\n'
                      'ctr = {}\n'
                      'epoch progress = {} \n'
                      'epoch = {}\n'
                      .format(data.train_index.ctr,
                              100. * data.train_index.ctr /
                              len(data.train_index.valid_data_moments),
                              epoch))

        data.train_index.epoch_complete = False

        mtl.save('save/mtl', epoch)
        direct.save('save/direct', epoch)
        follow.save('save/follow', epoch)
        furtive.save('save/furtive', epoch)
        control.save('save/control', epoch)

        while not data.val_index.epoch_complete:  # Epoch of training
            # Extract all data
            camera_data, metadata, target_data = batch.fill(data, data.val_index)
            dcamera, focamera, fucamera = camera_data.chunk(3, 0)
            dtarget, fotarget, futarget = target_data.chunk(3, 0)

            mtl.forward(camera_data, metadata, target_data)
            mtl.backward()

            control.forward(camera_data, target_data)
            control.backward()

            direct.forward(dcamera, dtarget)
            direct.backward()

            follow.forward(focamera, fotarget)
            follow.backward()

            furtive.forward(fucamera, futarget)
            furtive.backward()

            if print_counter.step(data.val_index):
                print('mode = train\n'
                      'ctr = {}\n'
                      'epoch progress = {} \n'
                      'epoch = {}\n'
                      .format(data.val_index.ctr,
                              100. * data.val_index.ctr /
                              len(data.val_index.valid_data_moments),
                              epoch))

        Utils.csvwrite('VALIDATION_LOSS.csv', [mtl.get_average(), 
                       direct.get_average(), follow.get_average(),
                       furtive.get_average(), control.get_average()])

        data.val_index.epoch_complete = False
        epoch += 1

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
