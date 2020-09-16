import torch
import torchvision
import torch.optim.lr_scheduler as ls
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from PIL import Image
from network import ReCoNet, Vgg16, gram,gram2, Normalization
from data_load import load_data, warp, warp2, get_mask2,MPIDataset2
import datetime
from tqdm import tqdm


#weight = 640
#height = 360
#weight = 384
#height = 216
weight = 512
#height = 218
height = 216

#LAMBDA_O = 1e6 ###1e6
#LAMBDA_F = 1e4 ##1e4
#ALPHA = 1e5  # previously 12, 2e10, 1e3
#BETA = 1e10  # 1e6 #11, 1e10
#GAMMA = 3e1  # previously 3e-2
#STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0]
STYLE_WEIGHTS = [1, 1e0, 1e0, 1e0]




def train2(args, data_train, data_test, model_style, model_loss, optimizer, schedular, device, style_img, mse_loss, msesum_loss,
          normalization_mean, normalization_std):
    #print('args:', args)

    normalization = Normalization(normalization_mean, normalization_std)

    style_img_list = []
    style_img_list.append(style_img)
#    model_style.train()
#    model_loss.eval()
    count = [0]
    while count[0] < args.epochs:
        data_bar=tqdm(data_train)
        count[0] += 1
        progress_num = [0]
        for id, sample in enumerate(data_bar):
            optimizer.zero_grad()

            img1, img2, mask, flow = sample
#            img1=img1.unsqueeze(0)
#            img2=img2.unsqueeze(0)
#            mask=mask.unsqueeze(0)
#            flow=flow.unsqueeze(0)
    
            #flow opitcal is contrary
            img1,img2=img2,img1
            img1 = img1.to(device)
            img2 = img2.to(device)
            mask = mask.to(device)
            flow = flow.to(device)

#            timet=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            img1_warp, mask_boundary_img1 = warp2(img1, flow,device)
            mask2 = get_mask2(img1_warp, img2, mask)
            

            style_img = style_img_list[0].to(device)
            feat1, output_img1 = model_style(img1)
            feat2, output_img2 = model_style(img2)

            #temp_feature_loss
            feat1_flow = torch.nn.Upsample(size=(feat1.size(2), feat1.size(3)), mode='bilinear',
                                           align_corners=True)(
                flow)
            mask_feat1 = torch.nn.Upsample(size=(feat1.size(2), feat1.size(3)), mode='bilinear',
                                           align_corners=True)(
                mask)
            feat1_warp, mask_boundary_feat1 = warp2(feat1, feat1_flow,device)
            temp_feature_loss = msesum_loss(feat2, feat1_warp)
            
            mask_feat1 = get_mask2(feat1_warp, feat2, mask_feat1)
            temp_feature_loss = torch.sum(temp_feature_loss * mask_feat1 * mask_boundary_feat1) / (
                    feat2.size(0) * feat2.size(1) * feat2.size(2) * feat2.size(3))

            ##temp_output_loss
            output_img1_warp,_ = warp2(output_img1, flow,device)

            temp_output_loss = msesum_loss(output_img2, output_img1_warp)

#            temp_output_loss = torch.sum(temp_output_loss * mask.view(temp_output_loss.size())) / (
            
#            img1_warp, mask_boundary_img1 = warp2(img1, flow,device)
#            mask2 = get_mask2(img1_warp, img2, mask)
#            mask2_save=mask2[0].detach().cpu()
#            mask2_save=torchvision.transforms.ToPILImage()(mask2_save)
#            mask2_save.save('./mask_normal/mask2_save2.jpg')
#            print('save2 is finish')

            temp_output_loss = torch.sum(temp_output_loss * mask2 * mask_boundary_img1) / (
                    img2.size(0) * img2.size(1) * img2.size(2) * img2.size(3))

            ## normalization to vgg16
            output_img1 = normalization(output_img1)
            output_img2 = normalization(output_img2)
            img1 = normalization(img1)
            img2 = normalization(img2)

            style_img = normalization(style_img.repeat(output_img1.size(0), 1, 1, 1))

            output1 = model_loss(output_img1)
            output2 = model_loss(output_img2)

            output_style = model_loss(style_img)

            output_content1 = model_loss(img1)
            output_content2 = model_loss(img2)

            ##content_loss
            # content_loss = (mse_loss(output_content1[2], output1[2]) + mse_loss(output_content2[2], output2[2])) / (
            #         output1[2].size(0) * output1[2].size(1) * output1[2].size(2) * output1[2].size(3))

            content_loss = mse_loss(output_content1[2], output1[2]) + mse_loss(output_content2[2], output2[2])
#            content_loss = mse_loss(output_content1[2], output1[2])

            ##style_loss
            style_loss = 0.0
            for i in range(len(output_style)):
                output_style_g = gram2(output_style[i])
                output1_g = gram2(output1[i])
                output2_g = gram2(output2[i])
                style_loss += STYLE_WEIGHTS[i]*(mse_loss(output_style_g, output1_g) + mse_loss(output_style_g, output2_g))

            ##reg_loss
            reg_loss = torch.sum(torch.abs(output_img1[:, :, :, :-1] - output_img1[:, :, :, 1:])) + torch.sum(
                torch.abs(
                    output_img1[:, :, :-1, :] - output_img1[:, :, 1:, :]))
            reg_loss =reg_loss+ (torch.sum(torch.abs(output_img2[:, :, :, :-1] - output_img2[:, :, :, 1:])) + torch.sum(
                torch.abs(
                    output_img2[:, :, :-1, :] - output_img2[:, :, 1:, :])))
            reg_loss /= output_img1.size(0)


            progress_num[0]+=img1.size(0)
            if id % args.log_interval == 0:
                #print(
                #    'Epoch:{} [{}/{}]{:.2f}% temp_feature_loss:{:.9f} temp_output_loss:{:.8f} content_loss:{:.2f} style_loss:{:.7f} reg_loss:{:.1f}'.format(
                #        count[0], progress_num[0], len(data_train.dataset),
                #        progress_num[0] / len(data_train.dataset) * 100.0, temp_feature_loss.item(),
                #        temp_output_loss.item(), content_loss.item(),
                #        style_loss.item(),
                #        reg_loss.item()))
                
                data_bar.set_description(
                    'Epoch:%d temp_feat_lo:%.7f temp_out_lo:%.7f cont_lo:%.2f style_lo:%.7f reg_lo:%.1f'
                        %(
                        count[0],
                        temp_feature_loss.item(),
                        temp_output_loss.item(),
                        content_loss.item(),
                        style_loss.item(),
                        reg_loss.item()))


            temp_feature_loss*=args.LAMBDA_F
            temp_output_loss*=args.LAMBDA_O
            content_loss*=args.ALPHA
            style_loss*=args.BETA
            reg_loss*=args.GAMMA
             
            loss = temp_feature_loss + temp_output_loss + content_loss  + style_loss  + reg_loss
            loss.backward()


            optimizer.step()
        schedular.step()


    if (args.save_model):
        if (not os.path.exists(args.save_directory)):
            os.mkdir(args.save_directory)
        time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if(!args.output_model):
            torch.save(model_style.state_dict(), os.path.join(args.save_directory, '%s.pt' % (args.phase)))
        else:
            torch.save(model_style.state_dict(), os.path.join(args.save_directory, '%s_%s.pt' % (args.phase,time_str)))


def main():
    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input  batch size for test (default: 1)')
    parser.add_argument('--phase', type=str, default='train',
                        help='train, test, predict(default:train)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epoch (default:100)')
    parser.add_argument('--path', type=str, default='.',
                        help='path')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval(default:1)')
    parser.add_argument('--lr',type=float,default=0.001,
                        help='learning rate(default:0.0001)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--LAMBDA-O', type=float, default=1e1,
                        help='output_temp_loss hyperparameter')
    parser.add_argument('--LAMBDA-F', type=float, default=1e-1,
                        help='feature_temp_loss hyperparameter')
    parser.add_argument('--ALPHA', type=float, default=1e0,
                        help='content_loss hyperparameter')
    parser.add_argument('--BETA', type=float, default=1e5,
                        help='style_loss hyperparameter')
    parser.add_argument('--GAMMA', type=float, default=1e-6,
                        help='reg_loss hyperparameter')
    parser.add_argument('--model-name', type=str, default='',
                        help='model name')
    parser.add_argument('--style-name', type=str, default='',
                        help='style image name')
    parser.add_argument('--output-model',type=bool,default=False,
                        help='True: generate output model name')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    args = parser.parse_args()
    #args.path='../../mpidataset'
    #args.style_name='style_images/vanGogh.jpg'
    #args.epochs = 1

   ## args.path = r'F:\DATASET\MPI-Sintel-complete'
    #args.path ='.'
    ##style_img = Image.open(os.path.join(args.path, args.style_name))
    style_img = Image.open(args.style_name)
    style_img = style_img.resize((weight, height),Image.BILINEAR)
    style_img = torchvision.transforms.ToTensor()(style_img)

    print('=====>Building Model')
    model_style = ReCoNet().to(device)
    model_loss = Vgg16().to(device)
    for param in model_loss.parameters():
        param.requires_grad = False

    mse_loss = torch.nn.MSELoss()
    msesum_loss = torch.nn.MSELoss(reduction='none')

    optimizer=optim.Adamax(model_style.parameters(),lr=args.lr)
    
    schedular = ls.MultiStepLR(optimizer, milestones=[8, 20], gamma=0.2)

    print('=====>Loading Data')
    #os.chdir(args.path)
    #dataset_train = load_data(os.path.join(args.path, 'train'))
    #dataset_test = load_data(os.path.join(args.path, 'test'))
    dataset_train = MPIDataset2(os.path.join(args.path,'training'))
    dataset_test = MPIDataset2(os.path.join(args.path,'test'))
    print('train len:', len(dataset_train))
    print('test len:', len(dataset_test))

    data_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
    data_test = DataLoader(dataset=dataset_test, batch_size=args.test_batch_size)

    #sample = dataset_train[0]
    #print(len(sample))
    
    print('args:', args)

    if args.phase == 'train' or args.phase == 'Train':
        # a=1
        train2(args, data_train, data_test, model_style, model_loss, optimizer, schedular, device, style_img, mse_loss, msesum_loss,
              cnn_normalization_mean, cnn_normalization_std)
    elif args.phase == 'finetune':
        print('finetune')
        model_style = ReCoNet()
        model_style.load_state_dict(
            torch.load(os.path.join(args.save_directory, args.model_name)))
        model_style=model_style.to(device)
        optimizer=optim.Adamax(model_style.parameters(),lr=args.lr)
        train2(args, data_train, data_test, model_style, model_loss, optimizer, schedular, device, style_img, mse_loss, msesum_loss,
              cnn_normalization_mean, cnn_normalization_std)


if __name__ == '__main__':
    start_time=datetime.datetime.now()
#     print('start time:',start_time.strftime('%Y/%m/%d %H:%M:%S'))

    main()
    end_time=datetime.datetime.now()
#     print('end time:',end_time.strftime('%Y/%m/%d %H:%M:%S'))
    print('cost time:',end_time-start_time)
    # test()
