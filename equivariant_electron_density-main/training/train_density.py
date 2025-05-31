import sys
import os
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from e3nn import o3
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from ase import Atoms
from ase.io import write
import time
import pandas as pd
from collections import defaultdict

from mydata.tools.Dataset import Dataset
from mydata.tools.data_loader import split_train_validation_test, collate_dicts
from mydata.dos.get_dos import dos_prepare,typeread,spd_dict_get
from mydata.dos.process import get_dos_features
from mydata.dos.process import split_data
from mydata.density.c_density import dataset_get

from model.CNN import CNN
# from model.Mat2Spec import Mat2Spec,compute_loss
from model.e3nn import e3nn
# from model.split import split
from mace import modules
from mace.modules.models import MACE
# from loss.dilate_loss import dilate_loss
from ase.data import chemical_symbols
import json
from collections import Counter
from sklearn.model_selection import KFold
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def get_structure_name(z):
    # 统计每个元素的数量
    element_counts = Counter(z.tolist())
    # 构建化学式表示
    chemical_formula = ""
    for element, count in element_counts.items():
        # 假设使用 periodic_table 字典来存储元素符号和对应的化学名
        element_symbol = chemical_symbols[int(element)]
        chemical_formula += f"{element_symbol}{count}"
    return chemical_formula

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def lossPerChannel(y_ml, y_target,
                   Rs=[(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]):
    err = y_ml - y_target
    loss_perChannel_list = np.zeros(len(Rs))
    normalization = err.sum() / err.mean()
    counter = 0
    for mul, l in Rs:
        if l == 0:
            temp_loss = err[:, :mul].pow(2).sum().abs() / normalization
        else:
            temp_loss = err[:, counter:counter + mul * (2 * l + 1)].pow(2).sum().abs() / normalization
        loss_perChannel_list[l] += temp_loss.detach().cpu().numpy()
        # pct_deviation_list[l]+=temp_pct_deviation.detach().cpu().numpy()
        counter += mul * (2 * l + 1)
    return loss_perChannel_list

def write_loss_to_csv(input_array,model_name,filename):
    try:
        # 读取现有的CSV文件
        df = pd.read_csv(filename)
    except FileNotFoundError:
        # 如果文件不存在，或者发生其他错误，创建一个新的DataFrame
        df = pd.DataFrame()
    # 根据data_num值进行排序
    first_two_columns = df.iloc[:, :2]
    # 对剩余的列进行排序
    df = df.iloc[:, 2:]
    # 将输入数组的第一个元素作为关键字
    key = str(input_array[0])
    # 如果关键字在第一列中存在，找到对应的列并写入数据
    if model_name == 'CNN':
        n = 1
    elif model_name == 'mace':
        n = 2
    # if key in df.columns:
    #     index = df.columns.get_loc(key)
    #     for i, column in enumerate(input_array[1:]):
    #         row_index = i * 3 + n
    #         df.loc[row_index, index] = column
    # else:
    for i, value in enumerate(input_array[1:]):
        row_index = i * 3 + n
        df.loc[row_index, str(input_array[0])] = value
    sorted_indices = np.argsort(np.array(df.columns.astype(int)))
    # 根据排序后的索引号移动DataFrame的整列
    df_columns = pd.concat([df.iloc[:, idx] for idx in sorted_indices], axis=1)
    # 合并结果
    df = pd.concat([first_two_columns, df_columns], axis=1)
    # 将DataFrame写入到CSV文件
    df.to_csv(filename, index=False)
    print(f"数组已写入到 {filename} 文件中")

def save_loss(sys_name,train_loss_all,val_loss_all,train_loss_all_scaled,val_loss_all_scaled,train_loss_all_w,val_loss_all_w,train_loss_all_feature,val_loss_all_feature,num_data,num_epochs,save_loss_pic,model_name):
    x = torch.arange(0,num_epochs)
    plt.plot(x,train_loss_all ,label='Train_Loss')
    plt.plot(x,val_loss_all ,label='Test_Loss')
    # 设置横轴标签和标题
    plt.xlabel('epoch')
    plt.xticks(torch.arange(0,num_epochs,20))
    plt.yticks(torch.arange(0.00, 0.25, 0.05))
    plt.title(f'{model_name}')
    ax = plt.gca()
    plt.text(0.45, 0.60, f'train_loss={min(train_loss_all):.4f}\nval_loss={min(val_loss_all):.4f}\n'
                         f'train_loss_scaled={min(train_loss_all_scaled):.4f}\nval_loss_scaled={min(val_loss_all_scaled):.4f}\n'
                         f'train_loss_w={min(train_loss_all_w):.4f}\nval_loss_w={min(val_loss_all_w):.4f}\n'
                         f'train_loss_feature={min(train_loss_all_feature):.4f}\nval_loss_feature={min(val_loss_all_feature):.4f}\n',
             transform=ax.transAxes, horizontalalignment='right')
    plt.legend()
    plt.savefig(save_loss_pic, format='jpg')
    plt.close()

    ## 把不同ecpoch写入loss_compare.csv文件中
    # loss_all = [num_data,min(train_loss_all).item(),min(val_loss_all).item(),
    #             min(train_loss_all_scaled).item(),min(val_loss_all_scaled).item(),
    #             min(train_loss_all_w).item(),min(val_loss_all_w).item(),
    #             min(train_loss_all_feature).item(),min(val_loss_all_feature).item()]
    # filename = f'../result/save_model/{sys_name}/loss_compare.csv'
    # write_loss_to_csv(loss_all, model_name,filename)

def wasserstein_distance(x, y, p=2):
    assert x.size() == y.size(), "Input tensors must have the same size"
    distance = (torch.abs(x - y) ** p).sum(dim=1)  # 计算每个样本的距离
    distance = torch.mean(distance ** (1/p))  # 取 p 次方根
    return distance



def test(model,device,test_loader,required_variable,energy_range,windows):
    with torch.no_grad():
        if required_variable=='dos_split':
            dos_save = torch.zeros([len(test_loader),2,12,windows,32])
        # z_save = []
        for test_dataset in [test_loader]:
            dos_loss_sum = 0.0
            dos_loss_scaled_sum = 0.0
            features_loss_sum = 0.0
            band_centers = 0.0
            band_widths = 0.0
            band_skews = 0.0
            band_kutosises = 0.0
            efs = 0.0

            dos_loss_scaled_all = []
            dos_loss_feature_all = []
            dos_ml_all = []
            dos_vasp_all = []
            n_all = []
            for i, data in enumerate(test_dataset):
                gpu_batch = dict()
                for key, val in data.items():
                    gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
                if required_variable == 'dos':
                    target_ml = model.forward(gpu_batch, required_variable)
                    for z in gpu_batch['z'].unique():
                        print(chemical_symbols[int(z)])
                    if os.path.exists(f"../result/dos/{i}")==False:
                        os.mkdir(f"../result/dos/{i}")
                    if os.path.exists(f"../result/dos/{i}/dos_ml.csv"):
                        os.remove(f"../result/dos/{i}/dos_ml.csv")
                    if os.path.exists(f"../result/dos/{i}/dos_vasp.csv"):
                        os.remove(f"../result/dos/{i}/dos_vasp.csv")
                    filename = f"../result/dos/{i}/POSCAR"
                    atoms = Atoms(symbols=gpu_batch['z'].cpu(), cell=gpu_batch['abc'].cpu(),positions=gpu_batch['pos'].cpu())
                    write(filename, atoms, format="vasp")
                    dos_ml = (target_ml['dos_ml']*target_ml['scaling'].unsqueeze(1)).cpu()
                    dos_vasp = (gpu_batch['dos']*gpu_batch['scaling'].unsqueeze(1)).cpu()
                    x = torch.linspace(-energy_range, energy_range, windows).to(target_ml['dos_ml'].device)

                    # 计算两个分布之间的Wasserstein距离
                    # P=target_ml['dos_ml'].detach().cpu().numpy() # 分布P，可以理解为分布P在X轴上的位置
                    # Q=gpu_batch['dos'].detach().cpu().numpy() # 同理
                    # for mm in range(P.shape[0]):
                    #     D1=scipy.stats.wasserstein_distance( x.cpu()+15, x.cpu()+15,abs(P[mm]), abs(Q[mm]))
                    #     # D1=scipy.stats.wasserstein_distance(dists,dists,P,Q)
                    #     print("Wasserstein距离为:", D1)


                    features_ml = get_dos_features(x, target_ml['dos_ml'])
                    features_true = get_dos_features(x, gpu_batch['dos'])
                    x = x.to('cpu')
                    features_loss = + getattr(F, "l1_loss")(features_ml, features_true)
                    # band_center = + getattr(F, "l1_loss")(features_ml[:, 0], features_true[:, 0])
                    # band_width = + getattr(F, "l1_loss")(features_ml[:, 1], features_true[:, 1])
                    # band_skew = + getattr(F, "l1_loss")(features_ml[:, 2], features_true[:, 2])
                    # band_kutosis = + getattr(F, "l1_loss")(features_ml[:, 3], features_true[:, 3])
                    # ef = + getattr(F, "l1_loss")(features_ml[:, 4], features_true[:, 4])
                    dos_loss_scaled = getattr(F, "l1_loss")((target_ml['dos_ml']), (gpu_batch['dos']))
                    dos_loss = getattr(F, "l1_loss")(target_ml['dos_ml']*target_ml['scaling'].unsqueeze(1), gpu_batch['dos']*gpu_batch['scaling'].unsqueeze(1))

                    np.savetxt((f"../result/dos/{i}/dos_ml.csv"), dos_ml, delimiter=",")
                    np.savetxt((f"../result/dos/{i}/dos_vasp.csv"), dos_vasp, delimiter=",")
                    for n in range(dos_ml.shape[0]):
                        features_ml = get_dos_features(x,dos_vasp[n].unsqueeze(0),spd=False)
                        features_true = get_dos_features(x,dos_ml[n].unsqueeze(0),spd=False)
                        features_loss_n = getattr(F, "l1_loss")(features_ml, features_true)
                        dos_loss_norm = getattr(F, "l1_loss")(dos_vasp[n], dos_ml[n])
                        if dos_loss_norm>0.7:
                            break
                        dos_loss_scaled_all.append(dos_loss_norm)
                        dos_loss_feature_all.append(features_loss_n)
                        dos_ml_all.append(dos_ml[n])
                        dos_vasp_all.append(dos_vasp[n])
                        n_all.append(gpu_batch['z'][n])
                        plt.plot(x, dos_vasp[n], label='vasp')
                        plt.plot(x, dos_ml[n], label='ml')
                        # 设置横轴标签和标题
                        plt.xlabel('energy')
                        plt.title(f'{chemical_symbols[int(gpu_batch["z"][n].item())]}')
                        ax = plt.gca()
                        plt.text(0.25, 0.85, f'LDOS={dos_loss_norm:.2f}\nfeature={features_loss_n:.2f}', transform=ax.transAxes, horizontalalignment='right')
                        plt.legend()
                        plt.savefig(f'../result/dos/{i}/{n}.jpg', format='jpg')
                        plt.close()
                    scaling_loss = getattr(F, "l1_loss")(target_ml['scaling'], gpu_batch['scaling'])
                    print(i)
                    print(f'scaling_loss:{scaling_loss}')
                    print(f'dos_loss_scaled:{dos_loss_scaled}')
                    print(f'dos_loss:{dos_loss}')
                    print(f'feature_loss:{features_loss}')
                    print('+++++++++++++++++++++++++')
                    dos_loss_sum += dos_loss
                    dos_loss_scaled_sum += dos_loss_scaled
                    features_loss_sum += features_loss
                    # band_centers += band_center
                    # band_widths += band_width
                    # band_skews += band_skew
                    # band_kutosises += band_kutosis
                    # efs += ef

                elif required_variable == 'dos_split':
                    for z in gpu_batch['z'].unique():
                        print(chemical_symbols[int(z)])
                    target_ml = model.forward(gpu_batch, required_variable)
                    #各个原子分波spd轨道的dos值loss
                    dos_ml_spd = (target_ml['dos_ml']*target_ml['scaling'].unsqueeze(1).repeat(1,windows,1)).cpu()
                    dos_vasp_spd = (gpu_batch['dos']*gpu_batch['scaling'].unsqueeze(1).repeat(1,windows,1)).cpu()
                    dos_loss_norm_spd = torch.mean(getattr(F, "l1_loss")(dos_vasp_spd, dos_ml_spd, reduction='none'),dim=1)
                    #计算每个原子每个分波dos的loss+feature的loss
                    x = torch.linspace(-energy_range, energy_range, windows)
                    features_ml = get_dos_features(x, target_ml['dos_ml'].cpu())
                    features_true = get_dos_features(x, gpu_batch['dos'].cpu())
                    features_loss_spd = getattr(F, "l1_loss")(features_ml, features_true, reduction='none')

                    # #创建存放所有结果的文件夹
                    # if os.path.exists(f"../result/dos/{i}")==False:
                    #     os.mkdir(f"../result/dos/{i}")
                    #
                    # #计算各个原子总dos值并存储到.csv文件
                    # dos_ml_spd = dos_ml_spd.cpu()
                    # dos_vasp_spd = dos_vasp_spd.cpu()
                    # dos_ml = torch.sum(dos_ml_spd,2)
                    # dos_vasp = torch.sum(dos_vasp_spd,2)
                    # np.savetxt((f"../result/dos/{i}/dos_ml.csv"), dos_ml, delimiter=",")
                    # np.savetxt((f"../result/dos/{i}/dos_vasp.csv"), dos_vasp, delimiter=",")
                    #
                    # #绘制各个原子的dos总和的ml_vasp对比图并存储
                    # for n in range(dos_ml.shape[0]):
                    #     features_ml = get_dos_features(x,dos_vasp[n].unsqueeze(0))
                    #     features_true = get_dos_features(x,dos_ml[n].unsqueeze(0))
                    #     features_loss = getattr(F, "l1_loss")(features_ml, features_true)
                    #     dos_loss_norm = getattr(F, "l1_loss")(dos_vasp[n], dos_ml[n])
                    #     plt.plot(x, dos_vasp[n], label='vasp')
                    #     plt.plot(x, dos_ml[n], label='ml')
                    #     # 设置横轴标签和标题
                    #     plt.xlabel('energy')
                    #     plt.title(f'{chemical_symbols[int(gpu_batch["z"][n].item())]}')
                    #     ax = plt.gca()
                    #     plt.text(0.25, 0.85, f'Lfeature={features_loss:.2f}\nLDOS={dos_loss_norm:.2f}', transform=ax.transAxes, horizontalalignment='right')
                    #     plt.legend()
                    #     plt.savefig(f'../result/dos/{i}/{n}.jpg', format='jpg')
                    #     plt.close()
                    # np.savetxt((f"../result/dos/{i}/dos_ml.csv"), dos_ml, delimiter=",")
                    # np.savetxt((f"../result/dos/{i}/dos_vasp.csv"), dos_vasp, delimiter=",")
                    #
                    # #绘制各个原子的各个轨道的dos的ml_vasp对比图并存储
                    # for n in range(dos_ml.shape[0]):
                    #     for spd in range(dos_ml_spd.shape[2]):
                    #         plt.plot(x, dos_vasp_spd[n][:,spd], label='vasp')
                    #         plt.plot(x, dos_ml_spd[n][:,spd], label='ml')
                    #         # 设置横轴标签和标题
                    #         plt.xlabel('energy')
                    #         plt.title(f'{chemical_symbols[int(gpu_batch["z"][n].item())]}')
                    #         ax = plt.gca()
                    #         plt.text(0.25, 0.55, f'Lfeature={features_loss_spd[n][spd]:.2f}\nLDOS={dos_loss_norm_spd[n][spd]:.2f}',
                    #                  transform=ax.transAxes, horizontalalignment='right')
                    #         plt.legend()
                    #         if os.path.exists(f"../result/dos/{i}/{n}") == False:
                    #             os.mkdir(f"../result/dos/{i}/{n}")
                    #         plt.savefig(f'../result/dos/{i}/{n}/{spd}.jpg', format='jpg')
                    #         plt.close()
                    #
                    #
                    # # 存储要用于计算后续吸附能的dos(大小为[2,400,104]->[ml+vasp,windows,(2*s+6*p+10*d)*3+(2*s+6*p)])
                    # dos_spd_save_ml = get_surface(gpu_batch['z'], dos_ml_spd.numpy(), energy_range, windows,spd=32)
                    # dos_spd_save_vasp = get_surface(gpu_batch['z'], dos_vasp_spd.numpy(), energy_range, windows,spd=32)
                    dos_save[i,:,:,:] = torch.cat((dos_ml_spd.unsqueeze(0),dos_vasp_spd.unsqueeze(0)),0)

                    # z_save.append(get_structure_name(gpu_batch['z']))
                    #进行分波dos与归一化后分波dos的loss计算并输出
                    dos_loss = getattr(F, "l1_loss")(target_ml['dos_ml'] * target_ml['scaling'].unsqueeze(1),
                                                     gpu_batch['dos'] * gpu_batch['scaling'].unsqueeze(1))
                    dos_loss_scaled = getattr(F, "l1_loss")((target_ml['dos_ml']), (gpu_batch['dos']))
                    print(i)
                    print(f'dos_scaled_loss:{dos_loss_scaled}')
                    print(f'dos_loss:{dos_loss}')
                    print('+++++++++++++++++++++++++')
                    dos_loss_sum += dos_loss
                    dos_loss_scaled_sum += dos_loss_scaled
            # if required_variable == 'dos_split':
            #     # np.savez(f'../result/dos/data.npz', dos=dos_save, z=z_save)
            #     np.savez(r'../dosnet/data.npz',dos = dos_save,z = z_save)

            # # 对scaled后的dos进行loss计算，并统计所有test_loader计算出的scaled_dos_loss分位数,绘制loss直方图
            # dos_loss_scaled_all = np.array(torch.Tensor(dos_loss_scaled_all))
            # dos_loss_feature_all = np.array(torch.Tensor(dos_loss_feature_all))
            # percentiles = np.percentile(dos_loss_scaled_all, [5, 25, 50, 75, 85])
            # bins = np.arange(0,0.2,0.001)
            # plt.hist(dos_loss_scaled_all, bins=bins, edgecolor='black',density=False)
            # plt.xlim(xmin=0, xmax=0.2)
            # plt.ylim(ymin=0, ymax=150)
            # plt.xlabel('Mean absolute error')
            # plt.ylabel('Count')
            # plt.title('Histogram of Data')
            # ax = plt.gca()
            # plt.text(0.75, 0.65, f"5th: {percentiles[0]:.4f}\n"
            #                      f"25th: {percentiles[1]:.4f}\n"
            #                      f"50th: {percentiles[2]:.4f}\n"
            #                      f"75th: {percentiles[3]:.4f}\n"
            #                      f"85th: {percentiles[4]:.4f}\n",
            #          transform=ax.transAxes,
            #          horizontalalignment='right')
            # # 在第25、50、75分位数处画竖线
            # for percentile in percentiles:
            #     plt.axvline(x=percentile, color='r', linestyle='--')
            # if os.path.exists(f'../result/compare/{model_name}/'):
            #     shutil.rmtree(f'../result/compare/{model_name}/')
            #     os.makedirs(f'../result/compare/{model_name}/', exist_ok=True)
            # else:
            #     os.makedirs(f'../result/compare/{model_name}/', exist_ok=True)
            # plt.savefig(f"../result/compare/{model_name}/Histogram.jpg", format='jpg')
            # plt.close()

            # # #对每个分位数提取出代表的dos绘制ml_vasp对比图
            # indices = []
            # p = 0
            # for percentile in percentiles:
            #     # 使用 np.where() 查找值的索引号
            #     indice = np.where(np.round(dos_loss_scaled_all, decimals=2) == np.round(percentile, decimals=2))[0]
            #     if os.path.exists(f'../result/compare/{model_name}/{per[p]}'):
            #         shutil.rmtree(f'../result/compare/{model_name}/{per[p]}')
            #         os.makedirs(f'../result/compare/{model_name}/{per[p]}', exist_ok=True)
            #     else:
            #         os.makedirs(f'../result/compare/{model_name}/{per[p]}', exist_ok=True)
            #     for i in indice:
            #         plt.plot(x, dos_vasp_all[i], label='vasp')
            #         plt.plot(x, dos_ml_all[i], label='ml')
            #         # 设置横轴标签和标题
            #         plt.xlabel('energy')
            #         plt.title(f'{chemical_symbols[int(n_all[i].item())]}')
            #         ax = plt.gca()
            #         plt.text(0.25, 0.85, f'LDOS={dos_loss_scaled_all[i]:.2f}\nfeature={dos_loss_feature_all[i]:.2f}', transform=ax.transAxes,
            #                  horizontalalignment='right')
            #         plt.legend()
            #         plt.savefig(f'../result/compare/{model_name}/{per[p]}/{i}.jpg', format='jpg')
            #         plt.close()
            #     indices.append(indice)
            #     p = p + 1
            # print("第5、25、50、75、85分位数值分别是：",percentiles,)
        print('平均：')
        print(f"feature预测误差{float(features_loss_sum/len(test_loader))}")
        print(f"band_center预测误差{float((band_centers)/len(test_loader))}")
        print(f"band_width预测误差{float((band_widths)/len(test_loader))}")
        print(f"band_skew预测误差{float((band_skews) / len(test_loader))}")
        print(f"band_kutosis预测误差{float((band_kutosises)/len(test_loader))}")
        print(f"Ef预测误差{float((efs)/len(test_loader))}")
        print(f"dos_scaled预测误差{float(dos_loss_scaled_sum)/len(test_loader)}")
        print(f"dos预测误差{float(dos_loss_sum/len(test_loader))}")



def main(args,config):
    print("详细参数值：")
    print(config)
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)
    torch.set_default_dtype(torch.float32)
    sys_name = args.sys_name.split('_')[0]
    required_variable = args.required_variable
    if os.path.exists('../result/save_model/' + sys_name + '/') == False:
        os.mkdir('../result/save_model/' + sys_name + '/')
    BATCH_SIZE = args.batch_size
    num_epochs = args.epochs
    model_name = args.model_name
    data_num = args.data_num
    data_start = args.data_start
    max_spd = 0
    patch_offsets = torch.IntTensor([[int(float(x)*args.patch_num) for x in y.split(',')] for y in args.patch_offset.split('-')])
    print(patch_offsets)
    seed = config["seed"]
    torch.manual_seed(seed)
    lr_initial = config["lr_initial"]
    r_cut = config["r_max"]
    if required_variable=='density' or required_variable=='density_spin':
        n_num = sum(1 for char in sys_name if char.isupper())
    elif required_variable=='dos' or required_variable=='dos_split':
        energy_range = config["range"]
        windows = config["windows"]
        if (sys_name == 'slab') | (sys_name =='surface') | (sys_name =='STO'):
            n_num = 40
        elif sys_name == 'TiOH':
            n_num = 40
        elif sys_name == 'bulk':
            n_num = 118

    if args.dataset_prepare.lower() == 'true':
        dataset_prepare = True
    else:
        dataset_prepare = False
    if args.pre_model.lower() == 'true':
        pre_model = True
    else:
        pre_model = False
    if args.use_sparse.lower() == 'true':
        use_sparse = True
    else:
        use_sparse = False
    if required_variable == 'density_spin':
        spin = True
    elif required_variable == 'density':
        spin = False
    elif required_variable == 'dos':
        split = False
        max_spd = 32
    elif required_variable == 'dos_split':
        split = True
        spd_dict = spd_dict_get(sys_name)
        max_spd = 0
        for v in spd_dict.values():
            count_ones = v.count(1)  # 计算当前列表中 1 的数量
            if count_ones > max_spd:
                max_spd = count_ones  # 更新最大值
    if dataset_prepare:
        sys.path.append('mydata/')
        print('----------------------正在准备数据-----------------------------')
        if required_variable=='density' or required_variable=='density_spin':
            dataset, z_table, m_l = dataset_get(sys_name=sys_name, density_num=args.data_num,
                                                patch_num=args.patch_num, n_num=n_num, r_cut=r_cut,
                                                data_start=data_start, spin=spin, patch_offsets=patch_offsets,
                                                use_sparse=use_sparse)
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            dataset = Dataset(dataset)
        elif required_variable=='dos' or required_variable == 'dos_split':
            dataset,z_table = dos_prepare(args.data_num, sys_name,args.model_name,n_num,energy_range,windows, save_pth=True,split = split,train_data_num=data_num)
        print('----------------------数据准备完成-----------------------------')
    else:
        if required_variable == 'dos_split':
            path = 'mydata/dos/' + sys_name + f'/data_analysis/{model_name}/{required_variable}_{data_num}.pth.tar'
            z_table = typeread(sys_name)
        elif required_variable == 'dos':
            if sys_name == 'slab':
                path = 'mydata/' + required_variable + '/' + sys_name + f'/data_analysis/{model_name}/{required_variable}_2500.pth.tar'
            elif sys_name == 'STO':
                path = 'mydata/' + required_variable + '/' + sys_name + f'/data_analysis/{model_name}/{required_variable}_5000.pth.tar'
            elif sys_name == 'TiOH':
                path = 'mydata/' + required_variable + '/' + sys_name + f'/data_analysis/{model_name}/{required_variable}_2000.pth.tar'
            else:
                path = 'mydata/' + required_variable + '/' + sys_name + f'/data_analysis/{model_name}/{required_variable}_{data_num}.pth.tar'
            z_table = typeread(sys_name)
        elif required_variable =='density' or required_variable == 'density_spin':
            n_num = sum(1 for char in args.sys_name if char.isupper())
            path = 'mydata/density/' + sys_name + f'/data_analysis/{required_variable}.pth.tar'
            z_table = typeread(sys_name)
            c_save_file = os.path.join('mydata/density/', sys_name, 'data_analysis/c.pth')
            if os.path.exists(c_save_file):
                c_save = torch.load(c_save_file)
                l_dict = c_save['l_dict']
                m_l = sum(list(l_dict.values()))
        elif required_variable =='density_spin':
            n_num = sum(1 for char in args.sys_name if char.isupper())
            path = 'mydata/density/' + sys_name + f'/data_analysis/{required_variable}.pth.tar'
            z_table = typeread(sys_name)
        dataset = Dataset(torch.load(path))
        # dataset = [dataset[i] for i in range(1000)]
        # 随机选取 data_num 个结构
        np.random.seed(seed)
        # dataset_indices = np.random.choice(len(dataset), data_num, replace=False)  # 随机选择 data_num 个索引
        dataset_indices = np.arange(data_num)
        print(dataset_indices)

    if sys_name=='slab' and (required_variable == 'dos' or required_variable == 'dos_split'):
        dataset_indices = np.arange(data_num - 42)
        last_indices = np.arange(len(dataset) - 42, len(dataset))
        dataset_indices = np.unique(np.concatenate((dataset_indices, last_indices)))

        dataset = [dataset[i] for i in dataset_indices]  # 根据索引选取子集

    #写入loss的result文件
    result_filename = f'../result/save_model/{sys_name}/{model_name}/{required_variable}_result.txt'
    if not os.path.exists(os.path.dirname(result_filename)):
        os.makedirs(os.path.dirname(result_filename))
    if os.path.exists(result_filename):
        os.remove(result_filename)
    ##Split datasets
    if sys_name == 'TiOH':
        kf = KFold(n_splits=min(10,len(dataset)), shuffle=False)  # 初始化KFold
    else:
        kf = KFold(n_splits=min(5,len(dataset)), shuffle=True,random_state=seed)  # 初始化KFold
    K = -1
    for train_index, val_index in kf.split(dataset):  # 调用split方法切分数据
        print(val_index)
        K = K + 1
        # if K >= 1 :
        #     break
        # 确保前 40 个数据始终在训练集中
        # fixed_indices = np.arange(37)  # 前 40 个数据的索引
        #train_index = np.concatenate((fixed_indices, train_index[train_index >= 37]))
        #val_index = val_index[val_index >= 37]  # 从验证集中移除前 40 个数据的索引
        with open(result_filename, 'a') as result_file:
            # 这里写入你想要在每个epoch写入的内容
            result_file.write(str(config)+'\n')
            result_file.write(str(args)+'\n')
            if required_variable == 'density' or required_variable == 'density_spin':
                result_file.write(" epoch|learning_rate|Train_DOS_Loss|Test_DOS_Loss|Train_DOS_LOSS_up|Test_DOS_LOSS_up|Train_DOS_LOSS_mag|Test_DOS_LOSS_mag|time\n"
                                  "---------------------------------------------\n")
            elif required_variable == 'dos' or required_variable == 'dos_split':
                result_file.write(" epoch|learning_rate|Train_DOS_Loss|Test_DOS_Loss|Train_DOS_LOSS_scaled|Test_DOS_LOSS_scaled|Train_DOS_w|Test_DOS_w|Train_feature_Loss|Test_feature_Loss|time\n"
                                  "---------------------------------------------\n")
        train_dataset = [dataset[i] for i in train_index]
        val_dataset = [dataset[i] for i in val_index]
        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_dicts,
                                  # sampler=RandomSampler(train_dataset)
                                  )
        val_loader = DataLoader(val_dataset,
                                 batch_size=BATCH_SIZE,
                                 collate_fn=collate_dicts, )



        if model_name == 'CNN':
            model_kwargs = {
                "data": dataset,
                "windows": windows,
                "required_variable": required_variable,
                "dim1": n_num,
                "dim2": 400,
                "pre_fc_count": 1,
                "gc_count": 9,
                "batch_norm": "True",
                "batch_track_stats": "True",
                "dropout_rate": config["drop_rate"],
            }
            model = CNN(**model_kwargs)
        elif model_name == "mace":
            default_kwargs = {
                "windows": 500,  # 或者提供一个合适的默认值
                "m_l": 107,  # 或者提供一个合适的默认值
            }
            if 'm_l' not in locals():
                m_l = default_kwargs.get("m_l", None)
            if 'windows' not in locals():
                windows = default_kwargs.get("windows", None)
            model_kwargs = {
                "r_max": config["r_max"],
                "num_bessel": config["num_radial_basis"],
                "num_polynomial_cutoff": config["num_cutoff_basis"],
                "max_ell": args.max_ell,
                "interaction_cls": modules.interaction_classes[args.interaction],
                "num_interactions": config["num_interactions"],
                "num_elements": n_num,
                "hidden_irreps": o3.Irreps(config["hidden_irreps"]),
                "avg_num_neighbors": config["avg_num_neighbors"],
                # "atomic_numbers": list(z_table.keys()),
                "correlation": config["correlation"],
                "gate": modules.gate_dict[config["gate"]],
                "interaction_cls_first": modules.interaction_classes["RealAgnosticInteractionBlock"],
                "dropout_rate": config["drop_rate"],
                # "conv_num": config["conv_num"],
                "required_variable": required_variable,
                "windows":  windows,
                "m_l": m_l,
                "max_spd":max_spd,
            }
            model = MACE(
                **model_kwargs,
            )
        elif model_name == "e3nn":
            default_kwargs = {
                "windows": 500,  # 或者提供一个合适的默认值
                "m_l": 107,  # 或者提供一个合适的默认值
            }
            if 'm_l' not in locals():
                m_l = default_kwargs.get("m_l", None)
            if 'windows' not in locals():
                windows = default_kwargs.get("windows", None)
            model_kwargs = {
                "in_num":n_num,
                "out_num":m_l,
                "irreps_in":config["irreps_in"],
                "irreps_hidden":config["irreps_hidden"],
                "irreps_out":config["irreps_out"],
                "irreps_node_attr":config["irreps_node_attr"],
                "irreps_edge_attr":config["irreps_edge_attr"],
                "layers":config["num_interactions"],
                "max_radius":config["r_max"],
                "number_of_basis":config["num_radial_basis"],
                "radial_layers":config["radial_layers"],
                "radial_neurons":config["radial_neurons"],
                "num_neighbors":config["avg_num_neighbors"],
                "num_nodes":config["num_nodes"],
                "dropout_rate": config["drop_rate"],
                "required_variable": required_variable,
                "windows": windows,
                "m_l": m_l,
                "max_spd": max_spd,
            }
            model = e3nn(
                **model_kwargs,
            )
        if pre_model == True:
            if required_variable == "density" or required_variable == 'density_spin':
                load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + 'density_spin_mag.pth'
                model.load_state_dict(torch.load(load_model_file), strict=False)
            elif required_variable == 'dos' or required_variable == 'dos_split':
                # load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + required_variable+'_split_'+str(K)+'.pth'
                load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + required_variable + '_' + str(K) + '.pth'
                model.load_state_dict(torch.load(load_model_file), strict=False)
        # if K >= 1:
        #     load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + required_variable + '.pth'
        #     model.load_state_dict(torch.load(load_model_file), strict=False)
        optim = torch.optim.Adam(model.parameters(), lr=lr_initial)
        scheduler = lr_scheduler.ReduceLROnPlateau(optim, factor=config["factor"], mode='min', min_lr=config["min_lr"],
                                                   patience=config["patience"], threshold=config["threshold"])
        optim.zero_grad()
        model.to(device)

        def count_parameters(model):
            # 统计PyTorch模型的参数数量。
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        num_params = count_parameters(model)
        print(f'当前的模型的参数量为：{num_params}')
        if required_variable == 'density' or required_variable == 'density_spin':
            print(" epoch|learning_rate|Train_DOS_Loss|Test_DOS_Loss|Train_DOS_LOSS_up|Test_DOS_LOSS_up|Train_DOS_LOSS_mag|Test_DOS_LOSS_mag|time")
        else:
            print(" epoch|learning_rate|Train_DOS_Loss|Test_DOS_Loss|Train_DOS_LOSS_scaled|Test_DOS_LOSS_scaled|Train_DOS_w|Test_DOS_w|Train_feature_Loss|Test_feature_Loss|time")
        train_loss_all = []
        val_loss_all = []
        train_loss_scaled_all = []
        val_loss_scaled_all = []
        train_loss_w_all = []
        val_loss_w_all = []
        train_feature_loss_all = []
        val_feature_loss_all = []
        a_time = time.time()
        for epoch in range(num_epochs):
            Train_Loss = 0.0
            Test_Loss = 0.0
            Train_Loss_up = 0.0
            Test_Loss_up = 0.0
            Train_Loss_mag = 0.0
            Test_Loss_mag = 0.0
            Train_Loss_scaled = 0.0
            Test_Loss_scaled = 0.0
            Train_Loss_w = 0.0
            Test_Loss_w = 0.0
            Train_Loss_feature = 0.0
            Test_Loss_feature = 0.0
            for step, data in enumerate(train_loader):
                gpu_batch = dict()
                for key, val in data.items():
                    gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
                if required_variable == 'density':
                    target_ml = model.forward(gpu_batch, required_variable,spin=False)
                    num_atoms = gpu_batch['num_a']
                    num_batch = len(num_atoms)
                    num_feas = data['d_idx']
                    fea_seg = torch.cat([torch.ones(int(num_feas)) * i for i, num_feas in enumerate(num_feas)]).to(device)
                    err = 0
                    train_chg_loss = 0
                    train_chg_loss_up = 0
                    train_chg_loss_mag = 0
                    for batch_i in range(num_batch):
                        density_ml_i = target_ml[torch.nonzero(fea_seg == batch_i).squeeze()]
                        density_i = gpu_batch['density'][torch.nonzero(fea_seg == batch_i).squeeze()]
                        err_sq_i = torch.sum(torch.abs(density_ml_i - density_i)) / gpu_batch['charge_num'][batch_i]
                        # err_sq_i_rmse = torch.sum((density_ml_i - density_i)) / gpu_batch['charge_num'][batch_i]
                        err += err_sq_i
                        train_chg_loss += err_sq_i
                        train_chg_loss_up += 0
                        train_chg_loss_mag += 0
                    err = err / num_batch
                    train_chg_loss = train_chg_loss / num_batch
                    train_chg_loss_up = train_chg_loss_up / num_batch
                    train_chg_loss_mag = train_chg_loss_mag / num_batch
                elif required_variable == 'density_spin':
                    target_ml = model.forward(gpu_batch, required_variable,spin=True)
                    num_atoms = gpu_batch['num_a']
                    num_batch = len(num_atoms)
                    num_feas = data['d_idx']
                    fea_seg = torch.cat([torch.ones(int(num_feas)) * i for i, num_feas in enumerate(num_feas)]).to(device)
                    err = 0
                    train_chg_loss = 0
                    train_chg_loss_up = 0
                    train_chg_loss_mag = 0
                    for batch_i in range(num_batch):
                        density_ml_i = target_ml[torch.nonzero(fea_seg == batch_i).squeeze()]
                        density_i = gpu_batch['density'][torch.nonzero(fea_seg == batch_i).squeeze()]
                        # CHG 中所有网格点的电荷密度总值/网格总数（NGXF × NGYF × NGZF）= 体系的价电子数（NELECT），单位为 e/Å³
                        err_sq_i_tot = torch.sum(torch.abs((density_ml_i[:, 0]+density_ml_i[:, 1])-(density_i[:,0]+density_i[:,1]))) / gpu_batch['charge_num'][batch_i]
                        err_sq_i_up = torch.sum(torch.abs(density_ml_i[:, 0]-density_i[:,0])) / gpu_batch['charge_num'][batch_i]
                        err_sq_i_down = torch.sum(torch.abs(density_ml_i[:, 1]-density_i[:,1])) / gpu_batch['charge_num'][batch_i]
                        err_sq_i_mag = torch.sum(torch.abs((density_ml_i[:, 0]-density_ml_i[:, 1])-(density_i[:,0]-density_i[:,1]))) / gpu_batch['charge_num'][batch_i]


                        # err_sq_i_tot = torch.sum(torch.abs(density_ml_i[:, 0]-density_i[:,0])) / gpu_batch['charge_num'][batch_i]
                        # err_sq_i_mag = torch.sum(torch.abs(density_ml_i[:, 1]-density_i[:,1])) / gpu_batch['charge_num'][batch_i]
                        # err_sq_i_up = torch.sum(torch.abs((density_ml_i[:, 0]-density_ml_i[:, 1])-(density_i[:,0]-density_i[:,1]))) / gpu_batch['charge_num'][batch_i]/2
                        # err_sq_i_down = torch.sum(torch.abs((density_ml_i[:, 0]+density_ml_i[:, 1])-(density_i[:,0]+density_i[:,1]))) / gpu_batch['charge_num'][batch_i]/2

                        #mag不再以用电荷密度做归一化
                        err += config['tot'] * err_sq_i_tot + config['spin'] * err_sq_i_up + config['spin'] * err_sq_i_down + config['mag'] * err_sq_i_mag
                        train_chg_loss += err_sq_i_tot
                        train_chg_loss_up += err_sq_i_up
                        train_chg_loss_mag += err_sq_i_mag
                    err = err / num_batch
                    train_chg_loss = train_chg_loss / num_batch
                    train_chg_loss_up = train_chg_loss_up / num_batch
                    train_chg_loss_mag = train_chg_loss_mag / num_batch

                elif required_variable == 'dos':
                    target_ml = model.forward(gpu_batch, required_variable,split=False)
                    diff_ml = torch.diff(target_ml['dos_ml'])
                    diff_vasp = torch.diff(gpu_batch['dos'])
                    diff_ml_2 = torch.diff(diff_ml)
                    diff_vasp_2 = torch.diff(diff_vasp)

                    output_cumsum = torch.cumsum(target_ml['dos_ml'], axis=1)
                    dos_cumsum = torch.cumsum(gpu_batch['dos'], axis=1)

                    train_dos_loss_scaled = getattr(F, "l1_loss")(target_ml['dos_ml'], gpu_batch['dos'])
                    # print(train_dos_loss_scaled)
                    # print(f"train_scaled_loss:{train_dos_loss_scaled}")
                    train_dos_loss_w = wasserstein_distance(target_ml['dos_ml'], gpu_batch['dos'])
                    # dos_loss = getattr(F, "l1_loss")(target_ml['dos_ml'], gpu_batch['dos'])
                    dos_loss = train_dos_loss_scaled + config["w"] * train_dos_loss_w

                    train_dos_loss = getattr(F, "l1_loss")(target_ml['dos_ml'] * target_ml['scaling'].unsqueeze(1),
                                                           gpu_batch['dos'] * gpu_batch['scaling'].unsqueeze(1))
                    diff_loss = getattr(F, "l1_loss")(diff_ml[1:-2], diff_vasp[1:-2])
                    diff_loss_2 = getattr(F, "l1_loss")(diff_ml_2[1:-2], diff_vasp_2[1:-2])
                    scaling_loss = getattr(F, "l1_loss")(target_ml['scaling'], gpu_batch['scaling'])
                    dos_cumsum_loss = getattr(F, "l1_loss")(output_cumsum, dos_cumsum)

                    x = torch.linspace(-energy_range, energy_range, windows).to(target_ml['dos_ml'].device)
                    features_ml = get_dos_features(x, target_ml['dos_ml'], spd=False)
                    features_true = get_dos_features(x, gpu_batch['dos'], spd=False)
                    features_loss = getattr(F, "l1_loss")(features_ml, features_true)
                    err = config["dos"] * dos_loss + config["scaling"] * scaling_loss \
                        + config["diff"] * diff_loss + config["diff2"] * diff_loss_2 \
                        + features_loss * config["feature"] \
                        # + dos_cumsum_loss * config["cumsum"] \
                    # print(config["dos"] * dos_loss + config["scaling"] * scaling_loss)
                    # print(diff_loss)
                    # print(diff_loss_2)
                    # print('+++++++++++++')

                elif required_variable == 'dos_split':
                    target_ml = model.forward(gpu_batch, required_variable,split=True)
                    # 由计算出来的（归一化分波dos*归一化系数）加和得到总dos
                    dos_ml_all = torch.sum(target_ml['dos_ml'] * target_ml['scaling'].unsqueeze(1), axis=2)
                    dos_vasp_all = torch.sum(gpu_batch['dos'] * gpu_batch['scaling'].unsqueeze(1), axis=2)
                    # 计算总dos的loss用于评估
                    train_dos_loss = getattr(F, "l1_loss")(dos_ml_all, dos_vasp_all)
                    # 归一化后分波dos
                    train_dos_loss_scaled = getattr(F, "l1_loss")(target_ml['dos_ml'], gpu_batch['dos'])

                    # # 把费米能级附近的权重加重
                    # train_dos_loss_scaled_important = getattr(F, "l1_loss")(target_ml['dos_ml'][:, 230:270, :],
                    #                                                         gpu_batch['dos'][:, 230:270, :])

                    train_dos_loss_w = wasserstein_distance(target_ml['dos_ml'], gpu_batch['dos'])
                    # dos_loss = train_dos_loss_scaled
                    dos_loss = config['w'] * train_dos_loss_w + train_dos_loss_scaled
                    # 分波dos的归一化值
                    scaling_loss = getattr(F, "l1_loss")(target_ml['scaling'], gpu_batch['scaling'])

                    # 归一化后分波dos的一阶diff与二阶diff
                    diff_ml = torch.diff(target_ml['dos_ml'])
                    diff_vasp = torch.diff(gpu_batch['dos'])
                    diff_ml_2 = torch.diff(diff_ml)
                    diff_vasp_2 = torch.diff(diff_vasp)
                    diff_loss = getattr(F, "l1_loss")(diff_ml, diff_vasp)
                    diff_loss_2 = getattr(F, "l1_loss")(diff_ml_2, diff_vasp_2)

                    # 归一化后分波dos的积分
                    output_cumsum = torch.cumsum(target_ml['dos_ml'], axis=1)
                    dos_cumsum = torch.cumsum(gpu_batch['dos'], axis=1)
                    output_cumsum = torch.nan_to_num(output_cumsum, nan=0)
                    dos_cumsum = torch.nan_to_num(dos_cumsum, nan=0)
                    dos_cumsum_loss = getattr(F, "l1_loss")(output_cumsum, dos_cumsum)

                    # 归一化后分波dos的feature（center, width, ef_states）
                    x = torch.linspace(-energy_range, energy_range, windows).to(target_ml['dos_ml'].device)
                    features_ml = get_dos_features(x, target_ml['dos_ml'])
                    features_true = get_dos_features(x, gpu_batch['dos'])
                    features_loss = getattr(F, "l1_loss")(features_ml, features_true)

                    # loss function
                    err = config["dos"] * dos_loss \
                          + config["diff"] * diff_loss + config["diff2"] * diff_loss_2 \
                          + config["scaling"] * scaling_loss \
                    # err = config["dos"] * dos_loss + config["scaling"] * scaling_loss \
                    #       + config["diff"] * diff_loss + config["diff2"] * diff_loss_2\
                    #       + features_loss * config["feature"] + dos_cumsum_loss * config["cumsum"]
                if required_variable == 'density' or required_variable == 'density_spin':
                    Train_Loss += train_chg_loss
                    Train_Loss_up += train_chg_loss_up
                    Train_Loss_mag += train_chg_loss_mag
                elif required_variable == 'dos' or required_variable == 'dos_split':
                    Train_Loss += train_dos_loss.mean().detach().abs()
                    Train_Loss_scaled += train_dos_loss_scaled.mean().detach().abs()
                    Train_Loss_w += train_dos_loss_w.mean().detach().abs()
                    Train_Loss_feature += features_loss.mean().detach().abs()
                err.mean().backward()
                optim.step()
                optim.zero_grad()
                scheduler.step(err)
                # 清理 GPU 缓存
                torch.cuda.empty_cache()

            #     pbar.update(1)
            # pbar.close()

            # now the test loop
            with torch.no_grad():
                for val_dataset in [val_loader]:
                    for step, data in enumerate(val_dataset):
                        gpu_batch = dict()
                        for key, val in data.items():
                            gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
                        if required_variable == 'density':
                            target_ml = model.forward(gpu_batch, required_variable,spin=False)
                            num_atoms = gpu_batch['num_a']
                            num_batch = len(num_atoms)
                            num_feas = data['d_idx']
                            fea_seg = torch.cat(
                                [torch.ones(int(num_feas)) * i for i, num_feas in enumerate(num_feas)]).to(device)
                            err = 0
                            for batch_i in range(num_batch):
                                density_ml_i = target_ml[torch.nonzero(fea_seg == batch_i).squeeze()]
                                density_i = gpu_batch['density'][torch.nonzero(fea_seg == batch_i).squeeze()]
                                err_sq_i = torch.sum(torch.abs(density_ml_i - density_i)) / gpu_batch['charge_num'][batch_i]
                                err += err_sq_i
                            val_chg_loss = err / num_batch
                            val_chg_loss_up = 0
                            val_chg_loss_mag = 0
                        elif required_variable == 'density_spin':
                            target_ml = model.forward(gpu_batch, required_variable, spin=True)
                            num_atoms = gpu_batch['num_a']
                            num_batch = len(num_atoms)
                            num_feas = data['d_idx']
                            fea_seg = torch.cat([torch.ones(int(num_feas)) * i for i, num_feas in enumerate(num_feas)]).to(device)
                            err_tot = 0
                            err_up = 0
                            err_mag = 0
                            for batch_i in range(num_batch):
                                density_ml_i = target_ml[torch.nonzero(fea_seg == batch_i).squeeze()]
                                density_i = gpu_batch['density'][torch.nonzero(fea_seg == batch_i).squeeze()]
                                err_sq_i_tot = torch.sum(torch.abs(density_ml_i - density_i)) / gpu_batch['charge_num'][batch_i]
                                err_sq_i_up = torch.sum(torch.abs(density_ml_i[:,0] - density_i[:,0])) / gpu_batch['charge_num'][batch_i]
                                err_sq_i_mag = torch.sum(torch.abs((density_ml_i[:, 0] - density_ml_i[:, 1]) - (density_i[:, 0] - density_i[:, 1]))) / gpu_batch['charge_num'][batch_i]
                                # err_sq_i_tot = torch.sum(torch.abs(density_ml_i[:, 0] - density_i[:, 0])) / \
                                #                gpu_batch['charge_num'][batch_i]
                                # err_sq_i_mag = torch.sum(torch.abs(density_ml_i[:, 1] - density_i[:, 1])) / \
                                #                gpu_batch['charge_num'][batch_i]
                                # err_sq_i_up = torch.sum(torch.abs(
                                #     (density_ml_i[:, 0] - density_ml_i[:, 1]) - (density_i[:, 0] - density_i[:, 1]))) / \
                                #               gpu_batch['charge_num'][batch_i] / 2
                                err_tot += err_sq_i_tot
                                err_up += err_sq_i_up
                                err_mag += err_sq_i_mag
                            val_chg_loss = err_tot / num_batch
                            val_chg_loss_up = err_up / num_batch
                            val_chg_loss_mag = err_mag / num_batch

                        elif required_variable == 'dos':
                            target_ml = model.forward(gpu_batch, required_variable,split=False)
                            val_dos_loss_scaled = getattr(F, "l1_loss")(target_ml['dos_ml'], gpu_batch['dos'])
                            # print(f"val_scaled_loss:{val_dos_loss_scaled}")
                            val_dos_loss_w = wasserstein_distance(target_ml['dos_ml'], gpu_batch['dos'])
                            val_dos_loss = getattr(F, "l1_loss")(
                                target_ml['dos_ml'] * target_ml['scaling'].unsqueeze(1),
                                gpu_batch['dos'] * gpu_batch['scaling'].unsqueeze(1))
                            x = torch.linspace(-energy_range, energy_range, windows).to(target_ml['dos_ml'].device)
                            features_ml = get_dos_features(x, target_ml['dos_ml'], spd=False)
                            features_true = get_dos_features(x, gpu_batch['dos'], spd=False)
                            val_feature_loss = getattr(F, "l1_loss")(features_ml, features_true)
                        elif required_variable == 'dos_split':
                            target_ml = model.forward(gpu_batch, required_variable,split=True)
                            # 由计算出来的（归一化分波dos*归一化系数）加和得到总dos
                            dos_ml_all = torch.sum(target_ml['dos_ml'] * target_ml['scaling'].unsqueeze(1), axis=2)
                            dos_vasp_all = torch.sum(gpu_batch['dos'] * gpu_batch['scaling'].unsqueeze(1), axis=2)
                            # 计算总dos的loss用于评估
                            val_dos_loss = getattr(F, "l1_loss")(dos_ml_all, dos_vasp_all)

                            # 归一化后分波dos
                            val_dos_loss_scaled = getattr(F, "l1_loss")(target_ml['dos_ml'], gpu_batch['dos'])

                            val_dos_loss_w = wasserstein_distance(target_ml['dos_ml'], gpu_batch['dos'])

                            # 归一化后分波dos的feature（center, width, ef_states）
                            x = torch.linspace(-energy_range, energy_range, windows).to(target_ml['dos_ml'].device)
                            features_ml = get_dos_features(x, target_ml['dos_ml'])
                            features_true = get_dos_features(x, gpu_batch['dos'])
                            val_feature_loss = getattr(F, "l1_loss")(features_ml, features_true)
                        if required_variable == 'density' or required_variable == 'density_spin':
                            Test_Loss += val_chg_loss
                            Test_Loss_up += val_chg_loss_up
                            Test_Loss_mag += val_chg_loss_mag
                        elif required_variable == 'dos' or required_variable == 'dos_split':
                            Test_Loss += val_dos_loss.mean().detach().abs()
                            Test_Loss_scaled += val_dos_loss_scaled.mean().detach().abs()
                            Test_Loss_w += val_dos_loss_w.mean().detach().abs()
                            Test_Loss_feature += val_feature_loss.mean().detach().abs()
            train_loss_all.append(float(Train_Loss / len(train_loader)))
            val_loss_all.append(float(Test_Loss / len(val_loader)))
            train_loss_scaled_all.append(float(Train_Loss_scaled / len(train_loader)))
            val_loss_scaled_all.append(float(Test_Loss_scaled / len(val_loader)))
            train_loss_w_all.append(float(Train_Loss_w / len(train_loader)))
            val_loss_w_all.append(float(Test_Loss_w / len(val_loader)))
            train_feature_loss_all.append(float(Train_Loss_feature / len(train_loader)))
            val_feature_loss_all.append(float(Test_Loss_feature / len(val_loader)))
            b_time = time.time()
            if epoch % 10 == 0:
                if required_variable == 'density' or required_variable == 'density_spin':
                    print(f"{str(epoch)}|{float(optim.state_dict()['param_groups'][0]['lr']):.8f}"
                          f"|{float(Train_Loss / len(train_loader)):.4f}|{float(Test_Loss / len(val_loader)):.4f}"
                          f"|{float(Train_Loss_up / len(train_loader)):.4f}|{float(Test_Loss_up / len(val_loader)):.4f}"
                          f"|{float(Train_Loss_mag / len(train_loader)):.4f}|{float(Test_Loss_mag / len(val_loader)):.4f}"
                          f"|{float(b_time - a_time):.2f}s\n")
                    with open(result_filename, 'a') as result_file:
                        # 这里写入你想要在每个epoch写入的内容
                        result_file.write(f"{str(epoch)}|{float(optim.state_dict()['param_groups'][0]['lr']):.8f}"
                                          f"|{float(Train_Loss / len(train_loader)):.4f}|{float(Test_Loss / len(val_loader)):.4f}"
                                          f"|{float(Train_Loss_up / len(train_loader)):.4f}|{float(Test_Loss_up / len(val_loader)):.4f}"
                                          f"|{float(Train_Loss_mag / len(train_loader)):.4f}|{float(Test_Loss_mag / len(val_loader)):.4f}"
                                          f"|{float(b_time - a_time):.2f}s\n")
                elif required_variable == 'dos' or required_variable == 'dos_split':
                    print(f"{str(epoch)}|{float(optim.state_dict()['param_groups'][0]['lr']):.8f}"
                          f"|{float(Train_Loss / len(train_loader)):.4f}|{float(Test_Loss / len(val_loader)):.4f}"
                          f"|{float(Train_Loss_scaled / len(train_loader)):.4f}|{float(Test_Loss_scaled / len(val_loader)):.4f}"
                          f"|{float(Train_Loss_w / len(train_loader)):.4f}|{float(Test_Loss_w / len(val_loader)):.4f}"
                          f"|{float(Train_Loss_feature / len(train_loader)):.4f}|{float(Test_Loss_feature / len(val_loader)):.4f}"
                          f"|{float(b_time - a_time):.2f}s\n")
                    with open(result_filename, 'a') as result_file:
                        # 这里写入你想要在每个epoch写入的内容
                        result_file.write(f"{str(epoch)}|{float(optim.state_dict()['param_groups'][0]['lr']):.8f}"
                                          f"|{float(Train_Loss / len(train_loader)):.4f}|{float(Test_Loss / len(val_loader)):.4f}"
                                          f"|{float(Train_Loss_scaled / len(train_loader)):.4f}|{float(Test_Loss_scaled / len(val_loader)):.4f}"
                                          f"|{float(Train_Loss_w / len(train_loader)):.4f}|{float(Test_Loss_w / len(val_loader)):.4f}"
                                          f"|{float(Train_Loss_feature / len(train_loader)):.4f}|{float(Test_Loss_feature / len(val_loader)):.4f}"
                                          f"|{float(b_time - a_time):.2f}s"
                                          "\n")
        save_dir = os.path.join('../result/save_model/', sys_name, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_loss_pic = os.path.join(save_dir, required_variable + '.jpg')
        save_model_file = os.path.join(save_dir, required_variable + f'_{K}.pth')
        # if required_variable == 'density' or required_variable == 'density_spin':
        #     save_model_file = os.path.join(save_dir, required_variable + f'.pth')
        # else:
        #     save_model_file = os.path.join(save_dir, required_variable + f'_{K}.pth')
        save_loss(sys_name,
                  torch.Tensor(train_loss_all), torch.Tensor(val_loss_all),
                  torch.Tensor(train_loss_scaled_all), torch.Tensor(val_loss_scaled_all),
                  torch.Tensor(train_loss_w_all), torch.Tensor(val_loss_w_all),
                  torch.Tensor(train_feature_loss_all), torch.Tensor(val_feature_loss_all),
                  data_num, num_epochs, save_loss_pic, model_name)
        if os.path.exists(save_model_file):
            os.remove(save_model_file)
        torch.save(model.state_dict(), save_model_file)
        print(save_model_file)


    # all_model_params = []
    # averaged_model_params = {}
    # if required_variable == 'dos' or required_variable == 'dos_split':
    #     for i in range(5):
    #         load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '_400/' + f'{required_variable}_{i}.pth'
    #         model.load_state_dict(torch.load(load_model_file), strict=False)
    #         model_params = {}
    #         for name, param in model.named_parameters():
    #             model_params[name] = param.data
    #         # 将当前模型的参数添加到列表中
    #         all_model_params.append(model_params)
    #     # 计算参数的平均值
    #     for name in all_model_params[0].keys():
    #         params = torch.stack([p[name] for p in all_model_params])
    #         averaged_model_params[name] = params.mean(0)
    #     # 将平均参数加载到模型中
    #     for name, param in model.named_parameters():
    #         param.data.copy_(averaged_model_params[name])
    #     save_dir = os.path.join('../result/save_model/', sys_name, model_name)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     save_model_file = os.path.join(save_dir, required_variable + '.pth')
    #     if os.path.exists(save_model_file):
    #         os.remove(save_model_file)
    #     torch.save(model.state_dict(), save_model_file)


        # load_model_file = '../result/save_model/' + sys_name + '/'+ model_name + '/' + required_variable + '.pth'
        # model.load_state_dict(torch.load(load_model_file))
        # model.eval()
        # test(model,device, test_loader, required_variable,energy_range,windows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train electron density')
    parser.add_argument('--sys_name', type=str)
    # parser.add_argument('--seed',type=int,default=30)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--required_variable', type=str,choices=['dos', 'dos_split','density','density_spin'],help='选择一个选项')
    parser.add_argument('--fine_tune', type=str,default='False')
    parser.add_argument('--pre_model', type=str,default='False')
    parser.add_argument('--dataset_prepare', type=str, default='False')
    parser.add_argument('--data_num',type=int)
    parser.add_argument('--data_start',type=int,default=0)
    parser.add_argument('--model_name', choices=['CNN', 'e3nn', 'mace','Mat2Spec_CNN','Mat2Spec_mace'], help='选择一个选项')
    parser.add_argument('--save_pth', type=str, default='True')

    #计算density的一些参数
    parser.add_argument('--patch_num', type=int,default=4)
    parser.add_argument('--POSCAR_only', type=str, default='False')
    parser.add_argument('--max_size', type=int, default=100000)
    parser.add_argument('--patch_offset',type=str,default='0,0,0')
    parser.add_argument('--use_sparse',type=str,default='False')

    #对于MACE可以设置以下参数
    parser.add_argument('--max_ell', help=r"highest \ell of spherical harmonics", type=int, default=3)
    parser.add_argument(
        "--interaction",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
        ],
    )
    # parser.add_argument(
    #     "--hidden_irreps",
    #     help="irreps for hidden node states",
    #     type=str,
    #     # default="32x0e + 32x1o + 32x2e + 32x3o",
    #     default="15x0e + 15x1o + 15x2e",
    # )
    # parser.add_argument(
    #     "--MLP_irreps",
    #     help="hidden irreps of the MLP in last readout",
    #     type=str,
    #     default="8x0e",
    # )
    args=parser.parse_args()
    configs = load_config('config.json')
    main(args,configs[args.required_variable][args.model_name][args.sys_name])