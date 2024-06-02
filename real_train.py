#-*- encoding: UTF-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import math
import sys
import random


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    setup_seed(seed=0)
    
    opt = TrainOptions().parse()
    dataset_train = create_dataset(opt.dataset_name, 'train', opt)
    dataset_size_train = len(dataset_train)
    print('The number of training images = %d' % dataset_size_train)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = ((model.start_epoch * (dataset_size_train // opt.batch_size)) \
                    // opt.print_freq) * opt.print_freq

    for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):
        # training
        epoch_start_time = time.time()
        epoch_iter = 0
        model.train()

        iter_data_time = iter_start_time = time.time()
        for i, data in enumerate(dataset_train):
            if total_iters % opt.print_freq == 0:
                t_data = time.time() - iter_data_time
            total_iters += 1
            epoch_iter += 1 
            model.set_input(data)
            model.optimize_parameters(epoch)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time)
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data, total_iters)
                if opt.save_imgs: # Too many images
                    visualizer.display_current_results(
                    'train', model.get_current_visuals(), total_iters)
                iter_start_time = time.time()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d'
                  % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %.3f sec'
              % (epoch, opt.niter + opt.niter_decay,
                 time.time() - epoch_start_time))
        model.update_learning_rate(epoch)

        sys.stdout.flush()
