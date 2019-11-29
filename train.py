import torch
import os
import time
from dataset import VOCDetection
from utils import SSDAugmentation
from config import voc, MEANS
from ssd import build_ssd
from layers.modules import MultiBoxLoss
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir="./log/second")

dataset_root = "./data/11.21/marked/"
learning_rate = 1e-4
momentum = 5e-4
weight_decay = 5e-4
gamma = 0.1
batch_size = 16
num_workers = 16
save_folder = "weights/"
basenet = "vgg16_reducedfc.pth"
start_iter = 0
use_cuda = torch.cuda.is_available()
resume = ""

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train():
    cfg = voc
    voc_dataset = VOCDetection(root=dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    # cuda parse
    net.to(device)
    # weight
    if resume:
        print('Resuming training, loading {}...'.format(resume))
        net.load_weights(resume)
    else:
        vgg_weights = torch.load(os.path.join(save_folder, basenet))
        # print(vgg_weights)
        # print(vgg_weights['features'])
        # for k in list(vgg_weights.keys()):
        #     _, new_key = k.split(".", 1)
        #     print("wzdebug: ", k, new_key)
        #     # print(vgg_weights[k])
        #     vgg_weights[new_key] = vgg_weights.pop(k)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, True)
    net.train()
    # loss counters
    # loc_loss = 0
    # conf_loss = 0
    # epoch = 0
    print('Loading the dataset...')

    # epoch_size = len(voc_dataset) // batch_size
    step_index = 0

    data_loader = data.DataLoader(voc_dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(start_iter, cfg['max_iter']):
        # if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
        #     update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
        #                     'append', epoch_size)
        #     # reset epoch loss counters
        #     loc_loss = 0
        #     conf_loss = 0
        #     epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        images = Variable(images.to(device))
        with torch.no_grad():
            targets = [ann.to(device) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # loc_loss += loss_l.data[0]
        # conf_loss += loss_c.data[0]
        # loc_loss += loss_l.data
        # conf_loss += loss_c.data

        writer.add_scalar('loss', loss.detach().cpu().numpy(), iteration)

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')

        # if args.visdom:
        #     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
        #                     iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 50 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(),
                       'weights/ssd300_VOC_' + repr(iteration) + '.pth')
    torch.save(net.state_dict(),
               save_folder + 'VOC' + '.pth')


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = learning_rate * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
