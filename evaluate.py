import time

import torch
import torch.utils.data as data
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import DataLoader


from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import warnings

warnings.filterwarnings('ignore')

def val(args,model=None,epoch=None,patch=False,patches=4, size=16, step=4, is_multi=False):
    set_seed(114514)
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if args.dataset_type == 'shanghai':
        test_folder = args.dataset_path + "/" + args.dataset_type + "/Test"
    else:
        test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers_test, drop_last=False)
    labels = np.load(args.dataset_path + '/label/frame_labels_' + args.dataset_type + '.npy')

    if model:
        model = model.cuda()
        model.eval()
    else:
        model = torch.load(args.model_dir)
        model.eval()

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])


    labels_list = []
    label_length = 0
    psnr_256 = {}
    psnr_multi = {}

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args.dataset_type == 'shanghai':
            labels_list = np.append(labels_list,labels[label_length + 4:videos[video_name]['length'] + label_length])       # shanghai
        else:
            labels_list = np.append(labels_list,labels[0][4 + label_length:videos[video_name]['length'] + label_length])  # ped2 and avenue
        label_length += videos[video_name]['length']
        psnr_256[video_name] = []
        psnr_multi[video_name] = []


    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    model.eval()

    for k, (datas) in enumerate(test_batch):
        if (k * args.test_batch_size) == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        imgs = datas.cuda() # 4,5,3,256,256
        pred_img, sim = model.forward(imgs[:, :12, :])

        if (patch):
            frame_five = imgs[:, 12:, :].squeeze(1)
            mse_pixel5 = (((pred_img + 1) / 2) - ((frame_five + 1) / 2)) ** 2
            mse5 = patch_max_mse(mse_pixel5.cpu().detach().numpy(), patches=patches, size=size, step=step,is_multi=is_multi)
            psnr_256[videos_list[video_num].split('/')[-1]].append(psnr(mse5))
        else:
            break

    psnr_256_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        '''normalize_score_list_gel                Max-min  regularization
        multi_future_frames_to_scores              Gaussian smoothing                 '''
        psnr_256_list.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_256[video_name]))))

    psnr_256_list = np.asarray(psnr_256_list)

    accuracy_256 = AUC(psnr_256_list, np.expand_dims(1 - labels_list, 0))
    return accuracy_256 * 100

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="MRP-GLC")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--save_epochs', type=int, default=5, help='number of epochs for training')
    parser.add_argument('--print_iter', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=100, help='number of the memory items')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_path', type=str, default='/home/zlab/disk1/hu/DATASET', help='directory of data')
    parser.add_argument('--RESUME', type=bool, default=False, help='directory of log')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--version', type=str, default='DCN_SAM_Fusion', help="oral:?????")
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--para', type=str, default="{p+0.00g,gamma=1,30_512}_{1}", help='??')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = torch.load('results/avenue/batch=8_lr=0.0002_memory=10,loss=int+grad/model.pth')
    args.dataset_type = "avenue"
    if args.dataset_type == 'ped2':
        auc = val(args, model, patch=True, patches=1, size=32, step=8, is_multi=True)
    else:
        auc = val(args, model, patch=True, patches=1, size=64, step=8, is_multi=True)
    print(auc)








