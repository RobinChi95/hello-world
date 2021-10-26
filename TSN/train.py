import argparse
import os
import numpy as np
import itertools
import datetime
import time
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model.RGnetmodel import *
from readyData import *
from utils import *
from model.SSIMloss import *

import torch.nn
import torch
import visdom



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--Parameter_saved", type=str, default="Rnet", help="file of parameters are saved after training")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--dataset", type=str, default="Pic_train", help="path of the train and test dateset")

opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.Parameter_saved, exist_ok=True)
os.makedirs("saved_models/%s" % opt.Parameter_saved, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
SSIM = SSIM()

cuda = torch.cuda.is_available()
print(cuda)
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
R_G = UNetPP()
R_D = Discriminator(input_shape)

if cuda:
    R_G = R_G.cuda()
    R_D = R_D.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_cycle = criterion_cycle.cuda()
    criterion_identity = criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    R_G.load_state_dict(torch.load("saved_models/%s/R_G_%d.pth" % (opt.Parameter_saved, opt.epoch)))
    R_D.load_state_dict(torch.load("saved_models/%s/R_D_%d.pth" % (opt.Parameter_saved, opt.epoch)))
else:
    # Initialize weights
    R_G.apply(weights_init_normal)
    R_D.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(R_G.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_R_D = torch.optim.Adam(R_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_R_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_R_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [transforms.ToTensor()]

# Training data loader
dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset, transforms_=transforms_, unaligned=False, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("%s" % opt.dataset, transforms_=transforms_, unaligned=False, mode="test"),
    batch_size=4,
    shuffle=False,
    num_workers=opt.n_cpu,
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    for j in range(len(val_dataloader)):
        imgs = next(iter(val_dataloader))
        R_G.eval()

        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = R_G(real_A)
        real_B = Variable(imgs["B"].type(Tensor))

        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        image_grid = torch.cat((real_A, fake_B, real_B), 1)
        save_image(image_grid, "images/%s/%s.JPEG" % (opt.Parameter_saved, batches_done), normalize=False)

# ----------
#  Training
# ----------

viz1 = visdom.Visdom(env='GAN-loss')
win1 = viz1.line(Y=np.array([0]), X=np.array([0]), opts={'title': 'GAN-loss'})
viz2 = visdom.Visdom(env='SSIM loss')
win2 = viz3.line(Y=np.array([0]), X=np.array([0]), opts={'title': 'SSIM loss'})
viz3 = visdom.Visdom(env='G loss')
win3 = viz4.line(Y=np.array([0]), X=np.array([0]), opts={'title': 'G loss'})
viz4 = visdom.Visdom(env='corre loss')
win4 = viz4.line(Y=np.array([0]), X=np.array([0]), opts={'title': 'corre loss'})

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))


        valid = Variable(Tensor(np.ones((real_A.size(0), *R_D.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *R_D.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        R_G.train()
        optimizer_G.zero_grad()

        fake_B = R_G(real_A)
        loss_GAN_AB = criterion_GAN(R_D(fake_B), valid)

        loss_GAN = loss_GAN_AB

        # corre loss
        loss_corre = criterion_GAN(real_B, R_G(real_A))

        ## SSIM loss
        loss_SSIM = SSIM(real_B,fake_B)

        # Total loss
        loss_G = 0.1*loss_GAN + 10 * loss_corre - 0.03*loss_SSIM

        loss_G.backward()
        optimizer_G.step()
        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_R_D.zero_grad()

        # Real loss
        loss_real = criterion_GAN(R_D(real_B), valid)

        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(R_D(fake_B_.detach()), fake)

        # Total Discriminator loss
        loss_R_D = (loss_real + loss_fake) / 2


        loss_R_D.backward()
        optimizer_R_D.step()
        loss_D = loss_R_D

        # Determine approximate time left
        batches_done = epoch * len(dataloader)
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        # print(" batches_done=",batches_done)
        print(" time_left= ", time_left)
        # Print loss
        loss = {
            "loss_GAN": loss_GAN.item(),
            "loss_SSIM": loss_SSIM.item(),
            "loss_G": loss_G.item(),
            "loss_corre": loss_corre.item()
        }

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
            )
        )
        viz1.line(Y=np.array([loss['loss_GAN']]), X=np.array([epoch]), update='append', win=win1,opts={'title': 'GAN-loss'})
        viz2.line(Y=np.array([loss['loss_SSIM']]), X=np.array([epoch]), update='append', win=win3,  opts={'title': 'SSIM loss'})
        viz3.line(Y=np.array([loss['loss_G']]), X=np.array([epoch]), update='append', win=win4,opts={'title': 'Gene loss'})
        viz4.line(Y=np.array([loss['loss_corre']]), X=np.array([epoch]), update='append', win=win6,opts={'title': 'corre loss'})

    sample_images(epoch)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_R_D.step()

    if opt.checkpoint_interval != -1 and (epoch) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(R_G.state_dict(), "saved_models/%s/R_G_%d.pth" % (opt.Parameter_saved, (epoch)))
        torch.save(R_D.state_dict(), "saved_models/%s/R_D_%d.pth" % (opt.Parameter_saved, (epoch)))