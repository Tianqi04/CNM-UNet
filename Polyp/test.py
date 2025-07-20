import torch
from torch.utils.data import DataLoader
import timm

from models.Con_UNet import Con_UNet
from dataloaders.POLYP_dataloader import POLYP_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from tensorboardX import SummaryWriter

from models.unet import UNet
from models.CNM_UNet import CNM_UNet
from utils_DUSE.convert import replace_bn_with_adabn

from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')

    work_dir = config.pretrained_path
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
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
    # Data Loading(validation)
    target_test_csv = []
    for target in config.Target_Dataset:
        target_test_csv.append(target + '_test.csv')

    ts_img_list, ts_label_list = convert_labeled_list(config.data_path, target_test_csv)
    target_test_dataset = POLYP_dataset(config.data_path, ts_img_list, ts_label_list,
                                        config.input_size_w)
    val_loader = DataLoader(dataset=target_test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=config.num_workers)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif config.network == 'Con_UNet':
        model = Con_UNet(n_channels=3, n_classes=1)
    elif config.network == 'CNM_UNet':
        model = CNM_UNet(activefunc='gelu', droprate=0, kernel_size=3, n_channels=3, n_classes=1)
    else:
        raise Exception('network in not right!')
    model = model.cuda()





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
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



    prompt = Prompt(prompt_alpha=config.prompt_alpha, image_size=config.input_size_h, batch_size=1).cuda()
    memory_bank = Memory(size=config.memory_size, dimension=prompt.data_prompt.numel())
    optimizer_DUSE = torch.optim.Adam(prompt.parameters(),
                                       lr=0.01,
                                       betas=(0.9, 0.99),
                                       weight_decay=0.00)
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)

        # Replace BN layer with AdaBn layer during testing phase
        total_replacements = replace_bn_with_adabn(model)
        log_info = f' DUSE strategy:'
        logger.info(log_info)
        loss = test_one_epoch_DUSE(
            val_loader,
            model,
            criterion, optimizer_DUSE,
            logger,
            config,
            prompt,
            memory_bank, device
        )
    else:
        print('no found best.pth!!!')


if __name__ == '__main__':
    config = setting_config
    main(config)