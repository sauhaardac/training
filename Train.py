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


def main():
    logging.basicConfig(filename='training.log', level=logging.DEBUG)
    logging.debug(ARGS)  # Log arguments

    # Set Up PyTorch Environment
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(ARGS.gpu)
    torch.cuda.device(ARGS.gpu)

    nets = []

    net1 = {'net':SqueezeNet().cuda(), 'criterion':torch.nn.MSELoss().cuda(),
            'val_loss':Utils.LossLog()}
    net1['optimizer'] = torch.optim.Adadelta(net1['net'].parameters())
    net1['net'].load_state_dict(torch.load('/hostroot/home/sauhaarda/working-directory/training-grounds/balance-dual-net/save/balance-dual-net/DFepoch7.weights'))
    nets.append(net1)

    net2 = {'net':SqueezeNetOrig().cuda(), 'criterion':torch.nn.MSELoss().cuda(),
            'val_loss':Utils.LossLog()}
    net2['optimizer'] = torch.optim.Adadelta(net1['net'].parameters())
    net2['net'].load_state_dict(torch.load('/hostroot/home/sauhaarda/working-directory/training-grounds/balance-dual-net/save/balance-dual-net/Depoch7.weights'))
    nets.append(net2)

    # if ARGS.resume_path is not None:
    #     for resume_file in resume_path:
    #         print('Resuming w/ ' + ARGS.resume_path, 'yellow')
    #         save_data = torch.load(ARGS.resume_path)
    #         net.load_state_dict(save_data)

    data = Data.Data()
    batch = Batch.Batch()

    # Maitains a list of all inputs to the network, and the loss and outputs for
    # each of these runs. This can be used to sort the data by highest loss and
    # visualize, to do so run:
    # display_sort_trial_loss(data_moment_loss_record , data)
    data_moment_loss_record = {}
    rate_counter = Utils.RateCounter()

    def run_net(data_index):
        batch.fill(data, data_index)  # Get batches ready
        batch.forward(optimizer, criterion, data_moment_loss_record)

    try:
        print_counter = Utils.MomentCounter(ARGS.print_moments)
        epoch = 0
        for net in nets:
            net['net'].eval()
        while not data.val_index.epoch_complete:
            camera_data, metadata, target_data = batch.fill(data, data.val_index)
            nets[0]['optimizer'].zero_grad()
            output = nets[0]['net'](Variable(camera_data), Variable(metadata)).cuda()
            loss = nets[0]['criterion'](output, Variable(target_data))
            nets[0]['val_loss'].add(loss.data[0])

            nets[1]['optimizer'].zero_grad()
            output = nets[1]['net'](Variable(camera_data)).cuda()
            loss = nets[1]['criterion'](output, Variable(target_data))
            nets[1]['val_loss'].add(loss.data[0])

            if print_counter.step(data.val_index):
                print('mode = validation\n'
                      'ctr = {}\n'
                      'epoch progress = {} %\n'
                      .format(data.val_index.ctr,
                              100. * data.val_index.ctr /
                              len(data.val_index.valid_data_moments)))

        for net in nets:
            logging.debug(net['val_loss'].average())
            print net['net']
            print net['val_loss'].average()
        
        data.val_index.epoch_complete = False
    except Exception:
        logging.error(traceback.format_exc())  # Log exception


if __name__ == '__main__':
    main()
