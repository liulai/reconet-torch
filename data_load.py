import io
import sys
import os
from python_pfm import read
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor, ToPILImage,Resize
from skimage import transform


#weight = 640
#height = 360

#weight = 384
#height = 216

weight = 512
#height = 218
height = 216


def normalize(input):
    return input


class MPIDataSet(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        sample = line.strip().split(' ')
        img1 = Image.open(sample[0]).resize((weight, height), Image.BILINEAR)
        img2 = Image.open(sample[1]).resize((weight, height), Image.BILINEAR)
        mask = Image.open(sample[2]).resize((weight, height), Image.BILINEAR)
        flow = read(sample[3])

        img1 = ToTensor()(img1).float()
        img2 = ToTensor()(img2).float()
        mask = ToTensor()(mask).float()
        h, w, c = flow.shape
        flow = torch.from_numpy(transform.resize(flow, (height, weight))).permute(2, 0, 1).float()
        flow[0, :, :] *= float(flow.shape[1] / h)
        flow[1, :, :] *= float(flow.shape[2] / w)
        
        ##take no occluded regions to compute
        mask = 1 - mask
        mask[mask < 0.99] = 0
        mask[mask > 0] = 1
        return img1, img2, mask, flow

def toString(num):
    string = str(num)
    while (len(string) < 4):
        string = "0" + string
    return string


class MPIDataset2(Dataset):

    def __init__(self, path, transform=None):
        """
        looking at the "clean" subfolder for images, might change to "final" later
        root_dir -> path to the location where the "training" folder is kept inside the MPI folder
        """
        # self.path = path + "training/"
        self.path = path
        self.transform = transform
        self.dirlist = os.listdir(self.path + "/clean/")
        self.dirlist.sort()
        # print(self.dirlist)
        self.numlist = []
        for folder in self.dirlist:
            self.numlist.append(len(os.listdir(self.path + "/clean/" + folder + "/")))

    def __len__(self):
        return sum(self.numlist) - len(self.numlist)

    def __getitem__(self, idx):

        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        for i in range(0, len(self.numlist)):
            folder = self.dirlist[i]
            path = self.path + "/clean/" + folder + "/"
            occpath = self.path + "/occlusions/" + folder + "/"
            flowpath = self.path + "/flow/" + folder + "/"
            if (idx < (self.numlist[i] - 1)):
                num1 = toString(idx + 1)
                num2 = toString(idx + 2)
                # img1 = io.imread(path + "frame_" + num1 + ".png")
                # img2 = io.imread(path + "frame_" + num2 + ".png")
                # mask = io.imread(occpath + "frame_" + num1 + ".png")
                # img1 = torch.from_numpy(transform.resize(img1, (weight, height))).to(device).permute(2, 0, 1).float()
                # img2 = torch.from_numpy(transform.resize(img2, (360, 640))).to(device).permute(2, 0, 1).float()
                # mask = torch.from_numpy(transform.resize(mask, (360, 640))).to(device).float()

                img1 = Image.open(path + "frame_" + num1 + ".png").resize((weight, height), Image.BILINEAR)
                img2 = Image.open(path + "frame_" + num2 + ".png").resize((weight, height), Image.BILINEAR)
                mask = Image.open(occpath + "frame_" + num1 + ".png").resize((weight, height), Image.BILINEAR)

                flow = read(flowpath + "frame_" + num1 + ".flo")
                # bilinear interpolation is default
                # originalflow = torch.from_numpy(flow)
                # flow = torch.from_numpy(transform.resize(flow, (360, 640))).to(device).permute(2, 0, 1).float()
                # flow[0, :, :] *= float(flow.shape[1]) / originalflow.shape[1]
                # flow[1, :, :] *= float(flow.shape[2]) / originalflow.shape[2]
                # print(flow.shape) #y,x,2
                # print(img1.shape)

                img1 = ToTensor()(img1).float()
                img2 = ToTensor()(img2).float()
                mask = ToTensor()(mask).float()
                h, w, c = flow.shape
                flow = torch.from_numpy(transform.resize(flow, (height, weight))).permute(2, 0, 1).float()
                flow[0, :, :] *= float(flow.shape[1] / h)
                flow[1, :, :] *= float(flow.shape[2] / w)

                ##take no occluded regions to compute
                mask = 1 - mask
                mask[mask < 0.99] = 0
                mask[mask > 0] = 1
                # return img1, img2, mask, flow
                break
            idx -= self.numlist[i] - 1
        if self.transform:
            # complete later
            pass
        # IMG2 should be at t in IMG1 is at T-1
        return (img1, img2, mask, flow)


def get_mask2(output, sample, mask):
    '''
    to retain unchanged places of current and previous images, the mask should eliminate change place
    :param output: optical flow image, B x C x H x W
    :param sample: previous image, B x C x H x W
    :param mask: occlusions image, block current image, B x 1 x H x W
    :return: B x 1 x H x W
    '''
    
    output_gray = 0.2989 * output[:, 2, :, :] + 0.5870 * output[:, 1, :, :] + 0.1140 * output[:, 0, :, :]
    sample_gray = 0.2989 * sample[:, 2, :, :] + 0.5870 * sample[:, 1, :, :] + 0.1140 * sample[:, 0, :, :]
    output_gray = output_gray.unsqueeze(1)
    sample_gray = sample_gray.unsqueeze(1)
    mask_contrary = torch.abs(output_gray - sample_gray)
    mask_contrary[mask_contrary < 0.05] = 0
    mask_contrary[mask_contrary > 0] = 1
    mask_contrary = mask - mask_contrary
    mask_contrary[mask_contrary < 0] = 0
    mask_contrary[mask_contrary > 0] = 1
    return mask_contrary

def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    datasets = MPIDataSet(lines)
    return datasets


def warp(img, flow,device):
    b, c, h, w = img.size()

    # mesh grid
    x = torch.arange(0, w)
    y = torch.arange(0, h)
    y, x = torch.meshgrid(y, x)
    gg = torch.cat((x.unsqueeze(0), y.unsqueeze(0))).repeat(b, 1, 1, 1).float().to(device)
    gg = gg + flow

    gg[:, 0, :, :] = 2.0 * gg[:, 0, :, :] / (w - 1) - 1.0
    gg[:, 1, :, :] = 2.0 * gg[:, 1, :, :] / (h - 1) - 1.0
    gg = gg.permute(0, 2, 3, 1)

    output = torch.nn.functional.grid_sample(img, gg)
    mask = torch.nn.functional.grid_sample(torch.ones(img.size(),device=device), gg, mode='bilinear')

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    output = output * mask

    return output



def warp2(img, flow, device):
    b, c, h, w = img.size()

    # mesh grid
    x = torch.arange(0, w)
    y = torch.arange(0, h)
    y, x = torch.meshgrid(y, x)
    gg = torch.cat((x.unsqueeze(0), y.unsqueeze(0))).repeat(b, 1, 1, 1).float().to(device)
    gg = gg + flow

    gg[:, 0, :, :] = 2.0 * gg[:, 0, :, :] / (w - 1) - 1.0
    gg[:, 1, :, :] = 2.0 * gg[:, 1, :, :] / (h - 1) - 1.0
    gg = gg.permute(0, 2, 3, 1)

    output = torch.nn.functional.grid_sample(img, gg)
    mask_boundary = torch.nn.functional.grid_sample(torch.ones(img.size(), device=device), gg, mode='bilinear')

    mask_boundary[mask_boundary < 0.9999] = 0
    mask_boundary[mask_boundary > 0] = 1

    output = output * mask_boundary
    return output, mask_boundary



def main():
    # data=MPIDataSet(r'F:\DATASET\MPI-Sintel-complete\training')
    os.chdir(r'F:\DATASET\MPI-Sintel-complete')
    data = load_data(r'F:\DATASET\MPI-Sintel-complete\train')
    sample = data[0]
    print(torch.max(sample[3]), torch.min(sample[3]))
    warp(sample[0].unsqueeze(0), sample[3].unsqueeze(0))

    img2=torch.nn.functional.interpolate(sample[3].unsqueeze(0),mode='bilinear',size=(100,200),align_corners=True)
    print(img2.size())

    plt.imshow(ToPILImage()(img2[0, 0]))
    plt.show()
    img=torch.nn.Upsample(size=(100,200),mode='bilinear',align_corners=True)(sample[3].unsqueeze(0))

    print(img.size())
    plt.imshow(ToPILImage()(img[0,0]))
    plt.show()
    plt.imshow(Resize((100,200),interpolation=Image.BILINEAR)(ToPILImage()(sample[3][0])))
    plt.show()


def test():
    a = torch.tensor([1, 2, 3, 4]).float().view(1, 1, 2, 2)
    x, y = torch.meshgrid(torch.arange(0, 3), torch.arange(0, 3))
    # print(x)
    gg = torch.cat((y.unsqueeze(0), x.unsqueeze(0))).unsqueeze(0)
    # print(gg.size())
    gg = 2.0 * gg / 3.0 - 1.0
    print(gg)

    gg = gg.permute(0, 2, 3, 1).float()
    # print(gg)
    print(torch.nn.functional.grid_sample(a, gg))
    print(a)


if __name__ == '__main__':
    main()
    # test()
