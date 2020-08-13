from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models



def label2onehot(labels):
        uni_labels = labels.unique(sorted=True)
        k = 0
        dic = {}
        for l in uni_labels:
            dic[str(l.item())] = k
            k += 1
        for (i, l) in enumerate(labels):
            labels[i] = dic[str(l.item())]
        return labels

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--n_class', type=int, default=10, help='n_class, default=10')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw', 'wikiart']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                   transforms.Resize((opt.imageSize+29)),
                                   transforms.FiveCrop((opt.imageSize+29)*(0.9)),
                                   transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(crop)) for crop in crops])),

    #dataset = dset.ImageFolder(root=opt.dataroot,
    #                           transform=transforms.Compose([
    #                               #transforms.Resize(opt.imageSize),
    #                               #transforms.CenterCrop(opt.imageSize),
    #                               transforms.ToTensor(),
    #                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               #transforms.FiveCrop(opt.imageSize*(0.9)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

print(train_size)
print(test_size)
print(len(dataset))


device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
#nz = int(opt.nz)
#ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self, ngpu, n_class):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.n_class = n_class
        self.main = nn.Sequential(
            #for 256x256 wikiart,
            # input is (nc) x 256 x 256 ndf = 64
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf/2) x 128 x 128
            nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.disc = nn.Sequential(
            nn.Linear(ndf*8*4*4,1),
            nn.Sigmoid()
        )
        self.clas = nn.Sequential(
            nn.Linear((ndf*8)*4*4, 1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, n_class)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            dis = nn.parallel.data_parallel(self.disc, output.view(-1,ndf*8*4*4), range(self.ngpu))
            cla = nn.parallel.data_parallel(self.clas,output.view(-1,ndf*8*4*4),range(self.ngpu))
        else:
            output = self.main(input)
            dis = self.disc(output.view(-1,ndf*8*4*4))
            cla = self.clas(output.view(-1,ndf*8*4*4))

        return dis.view(-1, 1).squeeze(1), cla#.view(self.n_class,1)
##class Discriminator(nn.Module):
#    def __init__(self, ngpu, n_class):
#        super(Discriminator, self).__init__()
#        self.ngpu = ngpu
#        self.n_class = n_class
#        self.main = nn.Sequential(
#            #for 256x256 wikiart,
#            # input is (nc) x 256 x 256
#            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf/2) x 128 x 128
#            nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf) x 64 x 64
#            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*2) x 32 x 32
#            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*4) x 16 x 16
#            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 8),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*8) x 8 x 8
#            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#            #nn.Sigmoid()
#        )
#        self.disc = nn.Sequential(
#            nn.Linear(ndf*8*8*8,1),
#            nn.Sigmoid()
#        )
#        self.clas = nn.Sequential(
#            nn.Linear((ndf*8)*8*8, 1024),
#            nn.LeakyReLU(0.2,inplace=True),
#            nn.Linear(1024, 512),
#            nn.LeakyReLU(0.2,inplace=True),
#            nn.Linear(512, n_class)
#        )
#
#    def forward(self, input):
#        if input.is_cuda and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#            dis = nn.parallel.data_parallel(self.disc, output.view(-1,ndf*8*8*8), range(self.ngpu))
#            cla = nn.parallel.data_parallel(self.clas,output.view(-1,ndf*8*8*8),range(self.ngpu))
#        else:
#            output = self.main(input)
#            dis = self.disc(output.view(-1,ndf*8*8*8))
#            cla = self.clas(output.view(-1,ndf*8*8*8))
#
#        return dis.view(-1, 1).squeeze(1), cla#.view(self.n_class,1)


n_class = opt.n_class


netD = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)
netD.fc = nn.Linear(2048, n_class)
netD = torch.nn.DataParallel(netD, range(ngpu))

netD.cuda()


#netD = Discriminator(ngpu,n_class).to(device)
#netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

b_loss = nn.BCELoss()
c_loss = nn.CrossEntropyLoss()

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        #real_cpu = data[0].to(device)
        bs, ncrops, c, h, w = data[0].size()
        real_cpu = (data[0].view(-1, c, h, w)).to(device)
        real_class_label = label2onehot(data[1]).to(device)
        batch_size = bs
        #label = torch.full((batch_size,), real_label, device=device)
        #real_r_out_navg,ireal_c_out_navg = netD(real_cpu)
        real_c_out_navg = netD(real_cpu)

        #print(real_r_out_navg.size(), real_c_out_navg.size())
        #real_r_out,real_c_out = real_r_out_navg.view(bs, ncrops, -1).mean(1), real_c_out_navg.view(bs, ncrops, -1).mean(1)
        real_c_out = real_c_out_navg.view(bs, ncrops, -1).mean(1)

        #real_r_out,real_c_out = netD(real_cpu)

        errD_real = c_loss(real_c_out, real_class_label)
        #print("5")
        errD_real.backward()
        #print("6")
        #D_x = real_r_out.mean().item()
        #D_class_x = real_c_out.mean().item()
        optimizerD.step()
        if (i%50 == 0):
            print('[%d/%d][%d/%d] Loss_D: %.4f' % (epoch, opt.niter, i, len(dataloader),
                 errD_real.item()))
    total=0
    correct=0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            #netD.zero_grad()
            #real_cpu = data[0].to(device)
            bs, ncrops, c, h, w = data[0].size()
            real_cpu = (data[0].view(-1, c, h, w)).to(device)
            real_class_label = label2onehot(data[1]).to(device)
            batch_size = bs
            #label = torch.full((batch_size,), real_label, device=device)
            #real_r_out_navg,real_c_out_navg = netD(real_cpu)
            real_c_out_navg = netD(real_cpu)

            #print(real_r_out_navg.size(), real_c_out_navg.size())
            #real_r_out,real_c_out = real_r_out_navg.view(bs, ncrops, -1).mean(1), real_c_out_navg.view(bs, ncrops, -1).mean(1)
            real_c_out = real_c_out_navg.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(real_c_out.data, 1)
            total += real_class_label.size(0)
            correct += (predicted == real_class_label).sum().item()
        print(total)
        print(correct)
        print('Accuracy of the network on the train images: %d %%' % (
                        100 * correct / total))
        total=0
        correct=0
        for i,tdata in enumerate(test_dataloader,0):
            bs, ncrops, c, h, w = tdata[0].size()
            treal_cpu = (tdata[0].view(-1, c, h, w)).to(device)
            #treal_cpu = tdata[0].to(device)

            treal_class_label = label2onehot(tdata[1]).to(device)

            outputs_navg = netD(treal_cpu)
            #print(treal_cpu.size())
            #print(outputs_navg.size())
            outputs = outputs_navg.view(bs,ncrops,-1).mean(1) 
            
            _, predicted = torch.max(outputs.data, 1)
            total += treal_class_label.size(0)
            correct += (predicted == treal_class_label).sum().item()
        print(total)
        print(correct)
        print('Accuracy of the network on the test images: %d %%' % (
                        100 * correct / total))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))


