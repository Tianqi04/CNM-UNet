import torch
from torch.utils.data import DataLoader
import timm
from tensorboardX import SummaryWriter

from models.Con_UNet import Con_UNet
from models.unet import UNet
from models.CNM_UNet import CNM_UNet
from utils_DUSE.convert import replace_bn_with_adabn
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
from utils_DUSE.config import Seg_loss
from utils_DUSE.metrics import calculate_metrics
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from dataloaders.transform import collate_fn_wo_transform, collate_fn_w_transform
import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    print('#----------Preparing dataset----------#')
    # Data Loading(training)
    source_train_csv = []
    if config.Source_Dataset != 'REFUGE_Valid':
        source_train_csv.append(config.Source_Dataset + '_train.csv')
        #source_train_csv.append(config.Source_Dataset + '_test.csv')
    else:
        source_train_csv.append(config.Source_Dataset + '.csv')
    sr_img_list, sr_label_list = convert_labeled_list(config.data_path, source_train_csv)
    train_dataset = OPTIC_dataset(config.data_path, sr_img_list, sr_label_list,
                                  config.input_size_w, img_normalize=False, batch_size=config.batch_size)
    print('Source Train Dataset: ', source_train_csv, len(train_dataset))
    train_loader= DataLoader(dataset=train_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          collate_fn=collate_fn_w_transform,
                                          num_workers=config.num_workers)

    # Data Loading(validation)
    target_test_csv = []
    for target in config.Target_Dataset:
        if target != 'REFUGE_Valid':
            #target_test_csv.append(target + '_train.csv')
            target_test_csv.append(target + '_test.csv')
        else:
            target_test_csv.append(target + '.csv')

    ts_img_list, ts_label_list = convert_labeled_list(config.data_path, target_test_csv)
    target_test_dataset = OPTIC_dataset(config.data_path, ts_img_list, ts_label_list,
                                        config.input_size_w, img_normalize=True)
    val_loader = DataLoader(dataset=target_test_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         drop_last=False,
                                         collate_fn=collate_fn_wo_transform,
                                         num_workers=config.num_workers)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'unet':
        model = UNet(n_channels=3, n_classes=2)
    elif config.network == 'Con_UNet':
        model = Con_UNet(n_channels=3, n_classes=2)
    elif config.network == 'CNM_UNet':
        model = CNM_UNet(activefunc='gelu', droprate=0, kernel_size=3, n_channels=3, n_classes=2)
    else:
        raise Exception('network in not right!')
    model = model.cuda()





    print('#----------Prepareing loss, opt, sch and amp----------#')
    #criterion = config.criterion
    lossmap = ['dice', 'bce']
    criterion = Seg_loss(lossmap)
    optimizer = get_optimizer(config, model,config.lr)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step,_ = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )
        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,device
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss2 = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,device
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)