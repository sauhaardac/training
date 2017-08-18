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

    net3 = SqueezeNet().cuda()
    criterion3 = torch.nn.MSELoss().cuda()
    optimizer3 = torch.optim.Adadelta(net3.parameters())

    # save_data = torch.load('save/DFepoch2.weights')
    # net1.load_state_dict(save_data)
    # save_data = torch.load('save/Depoch2.weights')
    # net2.load_state_dict(save_data)
    # save_data = torch.load('save/MDepoch2.weights')
    # net3.load_state_dict(save_data)

    data = Data.Data()
    batch = Batch.Batch(net1)
    rate_counter = Utils.RateCounter()

    try:
        epoch = 0
        logging.debug('Starting training epoch #{}'.format(epoch))

        net1.train()  # Train mode
        net2.train()  # Train mode
        net3.train()
        print_counter = Utils.MomentCounter(ARGS.print_moments)

        for batch_ctr in range(10000):  # Epoch of training
            # Extract all data
            camera_data, metadata, target_data = batch.fill(data, data.train_index, ('direct',), ('follow',))
            dcamera, fcamera = camera_data.chunk(2, 0)
            dmeta, fmeta = metadata.chunk(2, 0)
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

            # Forward Net3
            optimizer3.zero_grad()
            outputs3 = net3(Variable(dcamera), Variable(dmeta))
            loss3 = criterion3(outputs3, Variable(dtarget))
            loss3.backward()
            nnutils.clip_grad_norm(net3.parameters(), 1.0)
            optimizer3.step()


            if print_counter.step(data.train_index):
                print('mode = train\n'
                      'ctr = {}\n'
                      'net1 most recent loss = {}\n'
                      'net2 most recent loss = {}\n'
                      'net3 most recent loss = {}\n'
                      'epoch progress = {} \n'
                      'epoch = {}\n'
                      .format(data.train_index.ctr,
                              loss1.data[0],
                              loss2.data[0],
                              loss3.data[0],
                              100. * data.train_index.ctr /
                              len(data.train_index.valid_data_moments),
                              epoch))

        Utils.save_net('DFepoch{}'.format(epoch), net1)
        Utils.save_net('Depoch{}'.format(epoch), net2)
        Utils.save_net('MDepoch{}'.format(epoch), net3)

        net1.eval()
        net2.eval()
        net3.eval()

        log1 = Utils.LossLog()
        log2 = Utils.LossLog()
        log3 = Utils.LossLog()

        for batch_ctr in range(1000):  # Epoch of validation
            camera_data, metadata, target_data = batch.fill(data, data.train_index, ('direct',), ('direct',))

            # Forward Net1
            optimizer1.zero_grad()
            outputs1 = net1(Variable(camera_data), Variable(metadata)).cuda()
            loss1 = criterion1(outputs1, Variable(target_data))
            log1.add(loss1.data[0])

            # Forward Net2
            optimizer2.zero_grad()
            outputs2 = net2(Variable(dcamera)).cuda()
            loss2 = criterion2(outputs2, Variable(dtarget))
            log2.add(loss2.data[0])

            # Forward Net3
            optimizer3.zero_grad()
            outputs3 = net3(Variable(camera_data), Variable(metadata)).cuda()
            loss3 = criterion3(outputs3, Variable(target_data))
            log3.add(loss3.data[0])

            if print_counter.step(data.train_index):
                print('mode = train\n'
                      'ctr = {}\n'
                      'net1 most recent loss = {}\n'
                      'net2 most recent loss = {}\n'
                      'net3 most recent loss = {}\n'
                      'epoch progress = {} \n'
                      'epoch = {}\n'
                      .format(data.train_index.ctr,
                              loss1.data[0],
                              loss2.data[0],
                              loss3.data[0],
                              100. * data.train_index.ctr /
                              len(data.train_index.valid_data_moments),
                              epoch))

        print(log1.average())
        print(log2.average())
        print(log3.average())
        logging.debug('Net1 Loss: {} '.format(log1.average()))
        logging.debug('Net2 Loss: {} '.format(log2.average()))
        logging.debug('Net3 Loss: {} '.format(log3.average()))
        csvwrite('valloss.csv', [log1.average(), log2.average(), log3.average()])

    except Exception:
        # Interrupt Saves
        import traceback
        traceback.print_exc()
        Utils.save_net('interrupt_save', net1)


if __name__ == '__main__':
    main()
