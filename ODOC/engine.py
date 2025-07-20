import pdb
from configs.config_setting import setting_config
import numpy as np
from tqdm import tqdm
import torch
from thop import profile
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
import configs.config_setting as cfg
from dataloaders.normalize import normalize_image_to_0_1
from utils import save_imgs
from utils_DUSE.convert import AdaBN
from utils_DUSE.memory import Memory
from utils_DUSE.prompt import Prompt
from utils_DUSE.metrics import calculate_metrics
from torch.autograd import Variable
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer,
                    device):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    # VpTTA
    metrics_test = [[], [], [], []]
    metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data['data'], data['mask']
        images = torch.from_numpy(normalize_image_to_0_1(images)).to(dtype=torch.float32).to(device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32).to(device)

        if 'ege' in setting_config.network :
            gt_pre, out = model(images)
            loss = criterion(gt_pre, out, targets)
        else:
            out = model(images)
            loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('train_loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)


    scheduler.step(loss)
    return step, np.mean(loss_list)


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                  device):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data['data'], data['mask']
            img = torch.from_numpy(normalize_image_to_0_1(img)).to(dtype=torch.float32).to(device)
            msk = torch.from_numpy(msk).to(dtype=torch.float32).to(device)

            if 'ege' in setting_config.network:
                gt_pre, out = model(img)
                loss = criterion(gt_pre, out, msk)
            else:
                out = model(img)
                loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info,config.study_name)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,device,
                    test_data_name=None):
    model.eval()
    loss_list = []
    metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']
    metrics_test = [[], [], [], []]
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data['data'], data['mask']
            img = torch.from_numpy(img).to(dtype=torch.float32)
            msk = torch.from_numpy(msk).to(dtype=torch.float32)
            img, msk = Variable(img).to(device), Variable(msk).to(device)

            if 'ege' in setting_config.network:
                gt_pre, out = model(img)
                loss = criterion(gt_pre, out, msk)
            else:
                out = model(img)
                loss = criterion(out, msk)

            loss_list.append(loss.item())



            metrics = calculate_metrics(out.detach().cpu(), msk.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        print("test of best model, Test Metrics: ", print_test_metric_mean)
        print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
        log_info = (
            f"test of best model, Test Metrics: {print_test_metric_mean}, "
            f"Mean Dice: {(print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2}"
        )
        logger.info(log_info)

    return np.mean(loss_list)

def test_one_epoch_DUSE(test_loader,
                        model,
                        criterion, optimizer,
                        logger,
                        config,
                        prompt, memory_bank, device,
                        test_data_name=None):
    metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']
    preds = []
    gts = []
    loss_list = []
    metrics_test = [[], [], [], []]

    for i, data in enumerate(tqdm(test_loader)):
        img, msk = data['data'], data['mask']
        img = torch.from_numpy(img).to(dtype=torch.float32)
        msk = torch.from_numpy(msk).to(dtype=torch.float32)
        img, msk = Variable(img).to(device), Variable(msk).to(device)

        model.eval()
        prompt.train()

        flag = model.change_BN_status(new_sample=True)

        # Initialize Prompt
        if len(memory_bank.memory.keys()) >= config.neighbor:
            _, low_freq = prompt(img)
            init_data, score = memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=config.neighbor)
        else:
            init_data = torch.ones((1, 3, prompt.prompt_size, prompt.prompt_size)).data
        prompt.update(init_data)

        # Train Prompt for n iters (1 iter in our DUSE)
        for tr_iter in range(config.iters):
            prompt_x, _ = prompt(img)
            model(prompt_x)
            times, bn_loss = 0, 0
            for nm, m in model.named_modules():
                if isinstance(m, AdaBN):  # 0.5 200（0.5 30 30（50）） 200
                    weight = cfg.H_base
                    if nm == 'Mult_up.sigma2.0' or nm == 'Mult_up.sigma.1':
                        weight = cfg.k * cfg.H_base
                    bn_loss += weight * m.bn_loss
                    times += 1
            loss = bn_loss / times

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            flag = model.change_BN_status(new_sample=False)

        # Inference
        model.eval()
        prompt.eval()
        with torch.no_grad():
            prompt_x, low_freq = prompt(img)
            if 'ege' in setting_config.network:
                gt_pre, out = model(prompt_x)
                loss = criterion(gt_pre, out, msk)
            else:
                out = model(prompt_x)
                loss = criterion(out, msk)

        # Update the Memory Bank
        memory_bank.push(keys=low_freq.cpu().numpy(), logits=prompt.data_prompt.detach().cpu().numpy())
        metrics = calculate_metrics(out.detach().cpu(), msk.detach().cpu())
        for i in range(len(metrics)):
            assert isinstance(metrics[i], list), "The metrics value is not list type."
            metrics_test[i] += metrics[i]

        loss_list.append(loss.item())

    test_metrics_y = np.mean(metrics_test, axis=1)
    print_test_metric_mean = {}
    for i in range(len(test_metrics_y)):
        print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
    print("Test of the best model(DUSE), Test Metrics: ", print_test_metric_mean)
    print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)
    logger.info(f"Test of the best model(DUSE), Test Metrics: {print_test_metric_mean}")
    logger.info(f"Mean Dice: {(print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2}")

    return np.mean(loss_list)

