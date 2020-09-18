import datetime
import cv2
import os
import argparse
import torch
# import torchvision
import numpy as np
from network import ReCoNet
import sys


def video(path, path_img, device, output_dir='output.avi', fps=30, model=None, concat=False):
    im_dir = os.path.join(path, path_img)
    video_dir = os.path.join(path, output_dir)
    # fps = 30
    imgs_name = os.listdir(im_dir)
    imgs_name.sort()
    num = len(imgs_name)
    img = cv2.imread(os.path.join(im_dir, imgs_name[0]))
    h, w, _ = img.shape
    img_size = (w, h)
    if concat:
        img_size = (w, h * 2)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    start_time = datetime.datetime.now()
    print('start time:', start_time.strftime('%Y/%m/%d %H:%M:%S'))
    for i in range(len(imgs_name)):
        img_dir = os.path.join(im_dir, imgs_name[i])
        print(img_dir)
        frame = cv2.imread(img_dir)
        if model is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = torch.from_numpy(frame.astype('float32') / 255.0).permute(2, 0, 1)
            img = img.to(device)
            _, output = model(img.unsqueeze(0))
            concat_img = output.squeeze(0).permute(1, 2, 0)
            if concat:
                concat_img = torch.cat([img, output.squeeze(0)], dim=1).permute(1, 2, 0)
            concat_cv2 = concat_img.detach().cpu().numpy()

            frame = concat_cv2 * 255
            frame = frame.astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_writer.write(frame)

    video_writer.release()
    end_time = datetime.datetime.now()
    print('end time:', end_time.strftime('%Y/%m/%d %H:%M:%S'))
    print('cost time:', end_time - start_time)
    print('the video location is:', video_dir)
    print('finish')


def main():
    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('--fps', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--video-name', type=str, default='output\output.avi', metavar='N',
                        help='video-name')
    parser.add_argument('--mode', type=str, default='video', metavar='N',
                        help='video mode: video,video_style,concat')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--imgs-path', type=str, default='',
                        help='images path')

    parser.add_argument('--model-name', type=str, default='',
                        help='model name')

    args = parser.parse_args()
    path = '.'
    os.chdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReCoNet()
    model.load_state_dict(
        torch.load(os.path.join(args.save_directory, args.model_name), map_location=torch.device('cpu')))
    model = model.to(device)

    print(args)
    if args.mode == 'video':
        video(path, args.imgs_path, device, output_dir=args.video_name, fps=args.fps)

    if args.mode == 'video_style':
        video(path, args.imgs_path, device, output_dir=args.video_name, fps=args.fps, model=model, concat=False)

    if args.mode == 'concat':
        video(path, args.imgs_path, device, output_dir=args.video_name, fps=args.fps, model=model, concat=True)


if __name__ == '__main__':
    print('sys.argv:')
    for item in sys.argv:
        print(item, end=' ')
    print()

    main()
