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

    net1 = SqueezeNet().cuda()
    criterion1 = torch.nn.MSELoss().cuda()
    optimizer1 = torch.optim.Adadelta(net1.parameters())

    net2 = SqueezeNetOrig().cuda()
    criterion2 = torch.nn.MSELoss().cuda()
    optimizer2 = torch.optim.Adadelta(net2.parameters())

    if ARGS.resume_path is not None:
        cprint('Resuming w/ ' + ARGS.resume_path, 'yellow')
        save_data = torch.load(ARGS.resume_path)
        net.load_state_dict(save_data)

    data = Data.Data()
    batch = Batch.Batch(net1)

    # Maitains a list of all inputs to the network, and the loss and outputs for
    # each of these runs. This can be used to sort the data by highest loss and
    # visualize, to do so run:
    # display_sort_trial_loss(data_moment_loss_record , data)
    data_moment_loss_record = {}
    rate_counter = Utils.RateCounter()

    try:
        epoch = 0
        avg_train_loss = Utils.LossLog()
        avg_val_loss = Utils.LossLog()
        while True:
            logging.debug('Starting training epoch #{}'.format(epoch))

            net1.train()  # Train mode
            net2.train()  # Train mode
            epoch_train_loss = Utils.LossLog()
            print_counter = Utils.MomentCounter(ARGS.print_moments)

            while not data.train_index.epoch_complete:  # Epoch of training
                # Extract all data
                camera_data, metadata, target_data = batch.fill(data, data.train_index)
                dcamera, fcamera = camera_data.chunk(2, 0)
                dtarget, ftarget = target_data.chunk(2, 0)

                # Forward Net1
                optimizer1.zero_grad()
                outputs1 = net1(Variable(camera_data), Variable(metadata)).cuda()
                loss1 = criterion1(outputs1, Variable(target_data))
                loss1.backward()
                nnutils.clip_grad_norm(net1.parameters(), 1.0)
                optimizer1.step()

                # Forward Net2
                optimizer2.zero_grad()
                outputs2 = net2(Variable(dcamera)).cuda()
                loss2 = criterion2(outputs2, Variable(dtarget))
                loss2.backward()
                nnutils.clip_grad_norm(net2.parameters(), 1.0)
                optimizer2.step()

                if print_counter.step(data.train_index):
                    epoch_train_loss.export_csv(
                        'logs/epoch%02d_train_loss.csv' %
                        (epoch,))
                    print('mode = train\n'
                          'ctr = {}\n'
                          'net1 most recent loss = {}\n'
                          'net2 most recent loss = {}\n'
                          'epoch progress = {} \n'
                          'epoch = {}\n'
                          .format(data.train_index.ctr,
                                  loss1.data[0],
                                  loss2.data[0],
                                  100. * data.train_index.ctr /
                                  len(data.train_index.valid_data_moments),
                                  epoch))

                    if ARGS.display:
                        batch.display()
                        plt.figure('loss')
                        plt.clf()  # clears figure
                        print_timer.reset()

            data.val_index.epoch_complete = False
            Utils.save_net('DFepoch{}'.format(epoch), net1)
            Utils.save_net('Depoch{}'.format(epoch), net2)
            epoch += 1

    except Exception:
        # Interrupt Saves
        import traceback
        traceback.print_exc()
        Utils.save_net('interrupt_save', net1)


if __name__ == '__main__':
    main()
