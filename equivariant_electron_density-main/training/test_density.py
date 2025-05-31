import sys
import os
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3
import argparse
import os
import json
from ase.data import chemical_symbols
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler,SequentialSampler
import torch.nn.functional as F
from scipy import interpolate

from mydata.tools.Dataset import Dataset
from mydata.tools.data_loader import split_train_validation_test, collate_dicts
from mydata.dos.process import split_data
from mydata.dos.get_dos import dos_prepare,typeread,spd_dict_get
from mydata.dos.process import get_dos_features
from mydata.density.c_density import density_prepare

from model.CNN import CNN
from mace import modules
from mace.modules.models import MACE
from collections import Counter
import matplotlib.pyplot as plt
import shutil
import time
from sklearn.model_selection import KFold
import os
from scipy.ndimage import gaussian_filter1d
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def get_structure_name(z):
    # 统计每个元素的数量\
    element_counts = Counter(z.squeeze().int().tolist())
    # element_counts = torch.bincount(z.squeeze().int())
    # 构建化学式表示
    chemical_formula = ""
    for element, count in element_counts.items():
        # 假设使用 periodic_table 字典来存储元素符号和对应的化学名
        element_symbol = chemical_symbols[int(element)]
        chemical_formula += f"{element_symbol}{count}"
    return chemical_formula

def read_doscar(filename,energy_range,windows):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
        dos_n = []
        num = -1
        for d in data[5:]:
            if d.startswith('     20.00000000'):
            # if d.startswith('      1.34987607'):
                num = num + 1
                if num == 1:
                    dos_all = dos
                elif num > 1:
                    dos_n.append(dos)
                dos = []
            else:
                dos.append([float(x) for x in d.split()])
        dos_n.append(dos)
    if np.array(dos_n).shape[0]==1:
        dos_n = np.array(dos_n).transpose(0,2,1)[:,:9,].squeeze(0)
    else:
        dos_n = np.sum(np.array(dos_n).transpose(0, 2, 1)[:, :9, ],0)
    dos_C_spd_new = torch.zeros([windows, dos_n.shape[0] - 1])
    xfit = dos_n[0, :]
    for c in range(1, dos_n.shape[0]):
        yfit = dos_n[c, :]
        dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
        xnew = np.linspace(-energy_range, energy_range, windows)
        dos_C_spd_new[:, c - 1] = torch.Tensor(dos_fit(xnew))
    return dos_C_spd_new

def lossPerChannel(y_ml, y_target,Rs=[(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]):
    err = y_ml - y_target
    pct_dev = torch.div(err.abs(), y_target)
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

def CHG_vi(copy_file,result_file,copy,data_write):
    if copy != '0':
        if os.path.exists(result_file):
            os.remove(result_file)
        if os.path.exists(os.path.dirname(result_file)) == False:
            os.mkdir(os.path.dirname(result_file))
        shutil.copyfile(copy_file,result_file)
    with open(result_file, "a") as CHGCAR_ml:
        if copy != '0':
            CHGCAR_ml.write(copy)
        # 每10个数据添加一个换行符
        if data_write.shape[0] % 10 != 0:
            data_write_remain = data_write[-data_write.shape[0] % 10:]
            data_write = data_write[:-(data_write.shape[0] % 10)]
        lines = data_write.reshape(-1, 10)
        row_strings = [' '.join(map(str, line.tolist())) for line in lines]
        full_string = '\n'.join(row_strings) + '\n'
        if data_write.shape[0] % 10 != 0:
            full_string = full_string + ' '.join(data_write_remain)
        CHGCAR_ml.write(full_string)


def test_density(model, device, test_loader,DATANUM,POSCAR_only,sys_name,required_variable,vi_dict):
    with torch.no_grad():
        a = time.time()
        if required_variable == 'density':
            test_dataset = next(iter(test_loader))
            gpu_batch = dict()
            for key, val in test_dataset.items():
                gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
            target_ml = model.forward(gpu_batch, required_variable=required_variable,spin = False)
        elif required_variable == 'density_spin':
            if vi_dict['feature']==None:
                target_ml = torch.zeros(vi_dict['feature_shape'],2)
            else:
                test_dataset = next(iter(test_loader))
                gpu_batch = dict()
                for key, val in test_dataset.items():
                    gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
                target_ml = model.forward(gpu_batch, required_variable=required_variable,spin = True)

        num_atoms = vi_dict['num_atoms']
        num_batch = len(num_atoms)
        num_feas = vi_dict['d_idx']
        #可视化CHG
        if POSCAR_only:
            for batch_i in range(num_batch):
                if required_variable == 'density':
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHG",vi_dict['copy_line'],target_ml)
                elif required_variable == 'density_spin':
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'],target_ml[:, 0] + target_ml[:, 1])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHGCAR_mag", vi_dict['copy_line'], target_ml[:,0]-target_ml[:,1])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHGCAR_up", vi_dict['copy_line'], target_ml[:,0])


        else:
            fea_seg = torch.cat([torch.ones(int(num_feas)) * i for i, num_feas in enumerate(num_feas)]).to(device)
            err = 0
            err_up = 0
            err_mag = 0
            for batch_i in range(num_batch):
                if required_variable == 'density':
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHG",vi_dict['copy_line'],target_ml)
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/vasp/{DATANUM}/CHG", vi_dict['copy_line'],gpu_batch['density'])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml_vasp/{DATANUM}/CHG", vi_dict['copy_line'],target_ml - gpu_batch['density'])
                    density_ml_i = target_ml[torch.nonzero(fea_seg == batch_i).squeeze()]
                    density_i = gpu_batch['density'][torch.nonzero(fea_seg == batch_i).squeeze()]
                    err_sq_i = torch.sum(torch.abs(density_ml_i - density_i)) / gpu_batch['charge_num'][batch_i]
                    err += err_sq_i
                elif required_variable == 'density_spin':
                    #写入ML计算出的三个文件
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'], target_ml[:,0]+target_ml[:,1])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHGCAR_mag", vi_dict['copy_line'], target_ml[:,0]-target_ml[:,1])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml/{DATANUM}/CHGCAR_up", vi_dict['copy_line'], target_ml[:,0])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/ml/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'], target_ml[:,0])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/ml/{DATANUM}/CHGCAR_mag", vi_dict['copy_line'], target_ml[:,1])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/ml/{DATANUM}/CHGCAR_up", vi_dict['copy_line'], (target_ml[:,0]-target_ml[:,1])/2)
                    # 写入VASP计算出的三个文件
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/vasp/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'], gpu_batch['density'][:,0]+gpu_batch['density'][:,1])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/vasp/{DATANUM}/CHGCAR_mag",vi_dict['copy_line'], gpu_batch['density'][:,0]-gpu_batch['density'][:,1])
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/vasp/{DATANUM}/CHGCAR_up", vi_dict['copy_line'], gpu_batch['density'][:,0])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/vasp/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'], gpu_batch['density'][:,0])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/vasp/{DATANUM}/CHGCAR_mag",vi_dict['copy_line'], gpu_batch['density'][:,1])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/vasp/{DATANUM}/CHGCAR_up", vi_dict['copy_line'], (gpu_batch['density'][:,0]-gpu_batch['density'][:,1])/2)
                    # 写入ML-vasp计算出的三个文件
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml_vasp/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'], (target_ml[:,0]+target_ml[:,1])-(gpu_batch['density'][:,0]+gpu_batch['density'][:,1]))
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml_vasp/{DATANUM}/CHGCAR_mag",vi_dict['copy_line'], (target_ml[:,0]-target_ml[:,1])-(gpu_batch['density'][:,0]-gpu_batch['density'][:,1]))
                    CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                           f"../result/density/ml_vasp/{DATANUM}/CHGCAR_up", vi_dict['copy_line'], target_ml[:,0]-gpu_batch['density'][:,0])
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/ml_vasp/{DATANUM}/CHGCAR_tot", vi_dict['copy_line'],
                    #        (target_ml[:, 0]) - (gpu_batch['density'][:, 0]))
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/ml_vasp/{DATANUM}/CHGCAR_mag", vi_dict['copy_line'],
                    #        (target_ml[:, 1]) - (gpu_batch['density'][:, 1]))
                    # CHG_vi(f"data_to_be_predicted/density/{sys_name}/POSCAR_ALL/{DATANUM}/POSCAR",
                    #        f"../result/density/ml_vasp/{DATANUM}/CHGCAR_up",vi_dict['copy_line'], (target_ml[:,0]-target_ml[:,1])/2-(gpu_batch['density'][:,0]-gpu_batch['density'][:,1])/2)

                    density_ml_i = target_ml[torch.nonzero(fea_seg == batch_i).squeeze()]
                    density_i = gpu_batch['density'][torch.nonzero(fea_seg == batch_i).squeeze()]
                    err_sq_i_tot = torch.sum(torch.abs(density_ml_i - density_i)) / gpu_batch['charge_num'][batch_i]
                    err_sq_i_up = torch.sum(torch.abs(density_ml_i[:,0] - density_i[:,0])) / gpu_batch['charge_num'][batch_i]
                    err_sq_i_mag = torch.sum(torch.abs((density_ml_i[:, 0] - density_ml_i[:, 1]) - (density_i[:,0] - density_i[:,1]))) / gpu_batch['charge_num'][batch_i]
                    # err_sq_i_tot = torch.sum(torch.abs(density_ml_i[:,0] - density_i[:,0])) / gpu_batch['charge_num'][batch_i]
                    # err_sq_i_mag = torch.sum(torch.abs(density_ml_i[:, 1] - density_i[:, 1])) / gpu_batch['charge_num'][batch_i]
                    # err_sq_i_up = torch.sum(torch.abs((density_ml_i[:, 0] - density_ml_i[:, 1])/2 - (density_i[:,0] - density_i[:,1])/2)) / gpu_batch['charge_num'][batch_i]
                    err += err_sq_i_tot
                    err_up += err_sq_i_up
                    err_mag += err_sq_i_mag
            err = err / num_batch
            err_up = err_up / num_batch
            err_mag = err_mag / num_batch
            return err,err_up,err_mag

def windows_change(dos,en_range):
    if len(dos.shape) == 2:
        dos_new = torch.zeros([dos.shape[0], 400])
    else:
        dos_new = torch.zeros([dos.shape[0],400,dos.shape[2]])
    for i in range(dos.shape[0]):
        if len(dos.shape) == 2:
            xfit = np.linspace(-en_range, en_range, 500)
            yfit = torch.Tensor(gaussian_filter1d(dos[i, :], sigma=7))
            dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
            xnew = np.linspace(-en_range, en_range, 400)
            dos_new[i, :] = torch.Tensor(dos_fit(xnew))
        else:
            for spd in range(dos.shape[2]):
                xfit = np.linspace(-en_range, en_range, 500)
                yfit = torch.Tensor(gaussian_filter1d(dos[i, :,spd], sigma=7))
                dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
                xnew = np.linspace(-en_range, en_range, 400)
                dos_new[i, :,spd] = torch.Tensor(dos_fit(xnew))
    return dos

def plt_save(n,x,ml,vasp,save_file):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # 使用字体管理器设置 Arial 字体
    font_prop = FontProperties(family='Arial')
    vasp_color = (116 / 255, 167 / 255, 79 / 255)
    ml_color = (253 / 255, 128 / 255, 2 / 255)
    # ml_color = (143 / 255, 189 / 255, 219 / 255)
    # vasp_color = (253 / 255, 190 / 255, 167 / 255)
    # ml_color = '#2ca02c'
    # vasp_color = '#d62728'
    plt.figure(figsize=(9, 6))
    # 设置全局字体加粗
    plt.rcParams['font.weight'] = 'semibold'
    ax = plt.gca()
    linewidth = 3
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    # 设置横纵坐标的刻度线文字加粗并调整字体大小
    plt.tick_params(axis='x', labelsize=16, width=3,length=10, which='major', labelbottom=True, labelleft=True)
    plt.tick_params(axis='y', labelsize=16, width=3,length=10, which='major', labelbottom=True, labelleft=True)
    # 调整子图布局
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.2, top=0.85)
    # print(x.shape[0])
    # plt.plot(x, vasp, label='dft', color=(116 / 255, 167 / 255, 79 / 255), linewidth=2 * linewidth)
    # plt.plot(x, ml,label='predicted', color=(248 / 255, 130 / 255, 50 / 255), linewidth=2 * linewidth)
    # plt.fill_between(x, vasp, color=(116 / 255, 167 / 255, 79 / 255), alpha=0.2)

    plt.plot(x, vasp, label=f'{chemical_symbols[int(n)]} DFT', color=vasp_color, linewidth=linewidth)
    plt.plot(x, ml,label=f'{chemical_symbols[int(n)]} Predicted', color=ml_color, linewidth=linewidth)
    plt.fill_between(x, vasp, color=vasp_color, alpha=0.2)
    # plt.fill_between(x,  dos_ml_all[n] , color='yellow', alpha=0.2)
    # 设置横轴标签和标题
    plt.ylabel("DOS", fontsize=20, labelpad=15, fontweight='bold',fontproperties=font_prop)
    plt.xlabel('$E$ - $E_f$ (eV)', fontsize=20, labelpad=15, fontweight='bold', fontproperties=font_prop)
    # plt.title(f'{chemical_symbols[int(n)]}', fontsize=20, pad=15, fontweight='bold', fontproperties=font_prop)
    # plt.text(0.25, 0.85,
    #          f'LDOS={dos_loss_scaled_all[i]:.2f}\nfeature={dos_loss_feature_all[i]:.2f}',
    #          transform=ax.transAxes,
    #          horizontalalignment='right')

    # 显示图例（自动区分两条曲线）
    font_prop = FontProperties()
    font_prop.set_size(20)  # 设置字体大小为 19
    font_prop.set_family('Arial')  # 设置字体名称为 Arial

    # 显示图例，并应用字体属性
    plt.legend(prop=font_prop, frameon=False)

    # plt.show()
    if os.path.exists(os.path.dirname(save_file))== False:
        os.mkdir(os.path.dirname(save_file))
    plt.savefig(save_file, format='jpg')
    plt.close()

def test(sys_name,model,device,test_loader,required_variable,energy_range,windows,model_name,sns_num,val_index=None):
    with torch.no_grad():
        if required_variable=='dos_split':
            dos_save = torch.zeros([len(test_loader),2,12,windows,32])
        z_save = []
        for test_dataset in [test_loader]:
            dos_loss_sum = 0.0
            dos_loss_scaled_sum = 0.0
            features_loss_sum = 0.0
            band_centers = 0.0
            band_widths = 0.0
            band_skews = 0.0
            band_kutosises = 0.0
            efs = 0.0
            dos_loss_all = []
            dos_loss_scaled_all = []
            dos_loss_feature_all = []
            dos_ml_all = []
            dos_vasp_all = []
            n_all = []
            print('i|结构名称|dos_scaled_loss|dos_loss')
            data_sns = np.zeros([sns_num,sns_num])
            data_sns_scale = np.zeros([sns_num,sns_num])
            data_sns_scaled = np.zeros([sns_num, sns_num])
            data_n = np.zeros([sns_num, sns_num])
            for i, data in enumerate(test_dataset):

                gpu_batch = dict()
                for key, val in data.items():
                    gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
                if required_variable == 'dos':
                    target_ml = model.forward(gpu_batch, required_variable)
                    # 写入计算后的信息
                    if os.path.exists(f"../result/dos/{i}")==False:
                        os.mkdir(f"../result/dos/{i}")
                    else:
                        shutil.rmtree(f"../result/dos/{i}")
                        os.mkdir(f"../result/dos/{i}")
                    # if os.path.exists(f"../result/dos/{i}/dos_ml.csv"):
                    #     os.remove(f"../result/dos/{i}/dos_ml.csv")
                    # if os.path.exists(f"../result/dos/{i}/dos_vasp.csv"):
                    #     os.remove(f"../result/dos/{i}/dos_vasp.csv")
                    filename = f"../result/dos/{i}/POSCAR"
                    atoms = Atoms(symbols=gpu_batch['z'].cpu(), cell=gpu_batch['abc'].cpu(),positions=gpu_batch['pos'].cpu())
                    write(filename, atoms, format="vasp")
                    dos_ml = (target_ml['dos_ml']*target_ml['scaling'].unsqueeze(1)).cpu()
                    dos_vasp = (gpu_batch['dos']*gpu_batch['scaling'].unsqueeze(1)).cpu()
                    x = torch.linspace(-energy_range, energy_range, windows).to(target_ml['dos_ml'].device)

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

                    # np.savetxt((f"../result/dos/{i}/dos_ml.csv"), dos_ml, delimiter=",")
                    # np.savetxt((f"../result/dos/{i}/dos_vasp.csv"), dos_vasp, delimiter=",")
                    for n in range(dos_ml.shape[0]):
                        features_ml = get_dos_features(x,dos_vasp[n].unsqueeze(0),spd=False)
                        features_true = get_dos_features(x,dos_ml[n].unsqueeze(0),spd=False)
                        features_loss_n = getattr(F, "l1_loss")(features_ml, features_true)
                        dos_loss_no_norm = getattr(F, "l1_loss")(dos_vasp[n], dos_ml[n])
                        dos_loss_norm = getattr(F, "l1_loss")(target_ml['dos_ml'][n], gpu_batch['dos'][n])
                        # if dos_loss_norm>0.7:
                        #     break
                        dos_loss_all.append(dos_loss_no_norm)
                        dos_loss_scaled_all.append(dos_loss_norm)
                        dos_loss_feature_all.append(features_loss_n)
                        dos_ml_all.append(dos_ml[n])
                        dos_vasp_all.append(dos_vasp[n])
                        n_all.append(gpu_batch['z'][n])

                        # #可视化dos
                        plt_save(gpu_batch["z"][n].item(), x, dos_ml[n], dos_vasp[n], f'../result/dos/{i}/{n}.jpg')


                    scaling_loss = getattr(F, "l1_loss")(target_ml['scaling'], gpu_batch['scaling'])
                    print(i)
                    # if val_index is not None:
                    #     print(val_index[i])


                    if sys_name == 'slab':
                        z_table = typeread(sys_name)
                        # 提取 z_table 的值并排序
                        sorted_values = sorted(z_table.keys())
                        # 创建新字典，值作为键，索引作为值
                        sns_table = {value: index for index, value in enumerate(sorted_values)}
                        indice = [sns_table[x.item()] for x in gpu_batch['z'].unique()]
                        if len(gpu_batch['z'].unique()) == 1:
                            data_sns[indice[0], indice[0]] += dos_loss
                            data_sns_scale[indice[0], indice[0]] += scaling_loss
                            data_sns_scaled[indice[0], indice[0]] += dos_loss_scaled
                            data_n[indice[0], indice[0]] += 1
                        else:
                            data_sns[indice[0], indice[1]] += dos_loss
                            data_sns[indice[1], indice[0]] += dos_loss
                            data_sns_scale[indice[0], indice[1]] += scaling_loss
                            data_sns_scale[indice[1], indice[0]] += scaling_loss
                            data_sns_scaled[indice[0], indice[1]] += dos_loss_scaled
                            data_sns_scaled[indice[1], indice[0]] += dos_loss_scaled
                            data_n[indice[0], indice[1]] += 1
                            data_n[indice[1], indice[0]] += 1

                    print(f'scaling_loss:{scaling_loss}')
                    print(f'dos_loss_scaled:{dos_loss_scaled}')
                    print(f'dos_loss:{dos_loss}')
                    print(f'feature_loss:{features_loss}')
                    # print('+++++++++++++++++++++++++')
                    dos_loss_sum += dos_loss
                    dos_loss_scaled_sum += dos_loss_scaled
                    features_loss_sum += features_loss
                    # band_centers += band_center
                    # band_widths += band_width
                    # band_skews += band_skew
                    # band_kutosises += band_kutosis
                    # efs += ef

                elif required_variable == 'dos_split':
                    # for z in gpu_batch['z'].unique():
                    #     print(chemical_symbols[int(z)])
                    target_ml = model.forward(gpu_batch, required_variable,split=True)
                    #各个原子分波spd轨道的dos值loss
                    dos_ml_spd = (target_ml['dos_ml']*target_ml['scaling'].unsqueeze(1).repeat(1,windows,1)).cpu()
                    dos_vasp_spd = (gpu_batch['dos']*gpu_batch['scaling'].unsqueeze(1).repeat(1,windows,1)).cpu()
                    dos_loss_norm_spd = torch.mean(getattr(F, "l1_loss")(dos_vasp_spd, dos_ml_spd, reduction='none'),dim=1)
                    #计算每个原子每个分波dos的loss+feature的loss
                    x = torch.linspace(-energy_range, energy_range, windows)
                    features_ml = get_dos_features(x, target_ml['dos_ml'].cpu())
                    features_true = get_dos_features(x, gpu_batch['dos'].cpu())
                    features_loss_spd = getattr(F, "l1_loss")(features_ml, features_true, reduction='none')

                    #创建存放所有结果的文件夹
                    if os.path.exists(f"../result/dos/{i}")==False:
                        os.mkdir(f"../result/dos/{i}")
                    #
                    # #计算各个原子总dos值并存储到.csv文件
                    dos_ml_spd = dos_ml_spd.cpu()
                    dos_vasp_spd = dos_vasp_spd.cpu()
                    dos_ml = torch.sum(dos_ml_spd,2)
                    dos_vasp = torch.sum(dos_vasp_spd,2)
                    if windows == 500:
                        dos_ml_spd = windows_change(dos_ml_spd,energy_range)
                        dos_vasp_spd = windows_change(dos_vasp_spd, energy_range)
                        dos_ml = windows_change(dos_ml,energy_range)
                        dos_vasp = windows_change(dos_vasp, energy_range)
                    # np.savetxt((f"../result/dos/{i}/dos_ml.csv"), dos_ml, delimiter=",")
                    # np.savetxt((f"../result/dos/{i}/dos_vasp.csv"), dos_vasp, delimiter=",")
                    #
                    # 绘制各个原子的dos总和的ml_vasp对比图并存储
                    for n in range(dos_ml.shape[0]):
                        features_ml = get_dos_features(x,dos_vasp[n].unsqueeze(0))
                        features_true = get_dos_features(x,dos_ml[n].unsqueeze(0))
                        features_loss = getattr(F, "l1_loss")(features_ml, features_true)
                        dos_loss_norm = getattr(F, "l1_loss")(dos_vasp[n], dos_ml[n])
                        # plt_save(gpu_batch["z"][n].item(),x,dos_ml[n],dos_vasp[n],f'../result/dos/{i}/{n}.jpg')
                    # np.savetxt((f"../result/dos/{i}/dos_ml.csv"), dos_ml, delimiter=",")
                    # np.savetxt((f"../result/dos/{i}/dos_vasp.csv"), dos_vasp, delimiter=",")

                    # #绘制各个原子的各个轨道的dos的ml_vasp对比图并存储
                    # for n in range(dos_ml.shape[0]):
                    #     for spd in range(dos_ml_spd.shape[2]):
                    #         plt_save(gpu_batch["z"][n].item(), x, dos_ml_spd[n][:,spd], dos_vasp_spd[n][:,spd], f'../result/dos/{i}/{n}/{spd}.jpg')


                    dos_save[i,:,:,:] = torch.cat((dos_ml_spd.unsqueeze(0),dos_vasp_spd.unsqueeze(0)),0)
                    z_save.append(get_structure_name(gpu_batch['z']))
                    #进行分波dos与归一化后分波dos的loss计算并输出
                    dos_ml_all = torch.sum(target_ml['dos_ml'] * target_ml['scaling'].unsqueeze(1), axis=2)
                    dos_vasp_all = torch.sum(gpu_batch['dos'] * gpu_batch['scaling'].unsqueeze(1), axis=2)
                    # 计算总dos的loss用于评估
                    dos_loss = getattr(F, "l1_loss")(dos_ml_all, dos_vasp_all)

                    dos_loss_scaled = getattr(F, "l1_loss")((target_ml['dos_ml']), (gpu_batch['dos']))
                    # print(i)
                    # print(f'dos_scaled_loss:{dos_loss_scaled}')
                    # print(f'dos_loss:{dos_loss}')
                    # print('+++++++++++++++++++++++++')
                    print(f'{i}|{get_structure_name(gpu_batch["z"])}|{dos_loss_scaled}|{dos_loss}')
                    dos_loss_sum += dos_loss
                    dos_loss_scaled_sum += dos_loss_scaled

                    z_table = typeread('slab')
                    # 提取 z_table 的值并排序
                    sorted_values = sorted(z_table.keys())
                    # 创建新字典，值作为键，索引作为值
                    z_table = {value: index for index, value in enumerate(sorted_values)}
                    indice = [z_table[x.item()] for x in gpu_batch['z'].unique()]
                    if len(gpu_batch['z'].unique()) == 1 :
                        data_sns[indice[0],indice[0]] += dos_loss
                        data_sns_scaled[indice[0], indice[0]] += dos_loss_scaled
                        data_n[indice[0],indice[0]] += 1
                    else:
                        data_sns[indice[0],indice[1]] +=  dos_loss
                        data_sns[indice[1],indice[0]] += dos_loss
                        data_sns_scaled[indice[0],indice[1]] +=  dos_loss_scaled
                        data_sns_scaled[indice[1],indice[0]] += dos_loss_scaled
                        data_n[indice[0], indice[1]] += 1
                        data_n[indice[1], indice[0]] += 1

            if required_variable == 'dos_split':
                # np.savez(r'../dosnet/data.npz', dos=dos_save, z=z_save)
                # 检查文件是否存在
                if os.path.exists('../dosnet/data.npz'):
                    # 文件存在，加载现有数据
                    with np.load('../dosnet/data.npz') as data:
                        existing_data = {key: data[key] for key in data.files}

                    # 合并新数据和旧数据
                    existing_data['dos'] = np.concatenate((existing_data['dos'], dos_save))
                    existing_data['z'] = np.concatenate((existing_data['z'], z_save))

                    # 保存为新的.npz文件
                    new_file_path = '../dosnet/data.npz'
                    np.savez(new_file_path, **existing_data)

                else:
                    # 文件不存在，创建并保存新文件
                    np.savez('../dosnet/data.npz', dos=dos_save, z=z_save)

            #分数对比线
            if required_variable == 'dos' or required_variable== 'dos_split':
                # 对scaled后的dos进行loss计算，并统计所有test_loader计算出的scaled_dos_loss分位数,绘制loss直方图
                per = [5, 50, 85]
                dos_loss_scaled_all = np.array(torch.Tensor(dos_loss_scaled_all))
                dos_loss_all = np.array(torch.Tensor(dos_loss_all))
                dos_loss_feature_all = np.array(torch.Tensor(dos_loss_feature_all))
                percentiles = np.percentile(dos_loss_all, per)
                if sys_name == 'TiOH':
                    bins = np.arange(0, 0.3, 0.015)
                    plt.hist(dos_loss_scaled_all, bins=bins, edgecolor='black', density=False)
                    plt.xlim(xmin=0, xmax=0.3)
                    plt.ylim(ymin=0, ymax=10000)
                elif sys_name == 'slab':
                    bins = np.arange(0, 0.3, 0.02)
                    plt.hist(dos_loss_scaled_all, bins=bins, edgecolor='black', density=False)
                    plt.xlim(xmin=0, xmax=0.3)
                    plt.ylim(ymin=0, ymax=8000)
                elif sys_name == 'STO':
                    bins = np.arange(0, 0.15, 0.01)
                    plt.hist(dos_loss_scaled_all, bins=bins, edgecolor='black', density=False)
                    plt.xlim(xmin=0, xmax=0.1)
                    plt.ylim(ymin=0, ymax=10000)
                plt.xlabel('Mean absolute error')
                plt.ylabel('Count')
                plt.title('Histogram of Test err')
                ax = plt.gca()
                plt.text(0.75, 0.65, f"5th: {percentiles[0]:.4f}\n"
                # f"25th: {percentiles[1]:.4f}\n"
                f"50th: {percentiles[1]:.4f}\n"
                # f"75th: {percentiles[3]:.4f}\n"
                f"85th: {percentiles[2]:.4f}\n",
                         transform=ax.transAxes,
                         horizontalalignment='right')
                # 在第25、50、75分位数处画竖线
                for percentile in percentiles:
                    plt.axvline(x=percentile, color='r', linestyle='--')
                if os.path.exists(f'../result/compare/{model_name}/'):
                    shutil.rmtree(f'../result/compare/{model_name}/')
                    os.makedirs(f'../result/compare/{model_name}/', exist_ok=True)
                else:
                    os.makedirs(f'../result/compare/{model_name}/', exist_ok=True)
                plt.savefig(f"../result/compare/{model_name}/Histogram.jpg", format='jpg')
                plt.close()


                # #对每个分位数提取出代表的dos绘制ml_vasp对比图
                indices = []
                p = 0
                for percentile in percentiles:
                    # 使用 np.where() 查找值的索引号
                    indice = np.where(np.round(dos_loss_all, decimals=2) == np.round(percentile, decimals=2))[0]
                    # 对 dos_loss_all 进行排序，并获取排序后的索引
                    sorted_indice = np.argsort(dos_loss_scaled_all[indice])
                    indice = indice[sorted_indice]
                    # # 使用排序后的索引对 dos_loss_scaled_all 进行排序
                    # sorted_dos_loss_scaled_all = dos_loss_scaled_all[indice][sorted_indices]

                    if os.path.exists(f'../result/compare/{model_name}/{per[p]}'):
                        shutil.rmtree(f'../result/compare/{model_name}/{per[p]}')
                        os.makedirs(f'../result/compare/{model_name}/{per[p]}', exist_ok=True)
                    else:
                        os.makedirs(f'../result/compare/{model_name}/{per[p]}', exist_ok=True)
                    for i in indice[:min(1500,len(indice))]:
                        # #可视化dos
                        plt_save(n_all[i].item(), x, dos_ml_all[i], dos_vasp_all[i],f'../result/compare/{model_name}/{per[p]}/{i}.jpg')

                    indices.append(indice)
                    p = p + 1
                print("第5、25、50、75、85分位数值分别是：", percentiles, )


        print('平均：')
        print(f"feature预测误差{float(features_loss_sum/len(test_loader))}")
        print(f"band_center预测误差{float((band_centers)/len(test_loader))}")
        print(f"band_width预测误差{float((band_widths)/len(test_loader))}")
        print(f"band_skew预测误差{float((band_skews) / len(test_loader))}")
        print(f"band_kutosis预测误差{float((band_kutosises)/len(test_loader))}")
        print(f"Ef预测误差{float((efs)/len(test_loader))}")
        print(f"dos_scaled预测误差{float(dos_loss_scaled_sum)/len(test_loader)}")
        print(f"dos预测误差{float(dos_loss_sum/len(test_loader))}")
        return data_sns,data_sns_scale,data_sns_scaled,data_n

def main(args,config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    sys_name = args.sys_name
    required_variable = args.required_variable
    DATANUM = args.data_num
    max_size = args.max_size
    model_name = args.model_name
    patch_num = args.patch_num
    r_cut = config["r_max"]
    seed = config["seed"]
    density_spacing = 0.1
    if args.POSCAR_only.lower() == 'true':
        POSCAR_only = True
    else:
        POSCAR_only = False
    if args.dataset_prepare.lower() == 'true':
        dataset_prepare = True
    else:
        dataset_prepare = False
    if args.feature_filter.lower() == 'true':
        feature_filter = True
    else:
        feature_filter = False
    if required_variable == 'density_spin':
        spin = True
    elif required_variable == 'density':
        spin = False
    elif required_variable == 'dos':
        split = False
        spd_dict = None
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
        if required_variable == 'density' or required_variable == 'density_spin':
            n_num = sum(1 for char in args.sys_name if char.isupper())
            z_table = typeread(sys_name)
        elif (required_variable == 'dos') | (required_variable == 'dos_split'):
            energy_range = config["range"]
            windows = config["windows"]
            if (sys_name.startswith('slab')) | (sys_name == 'surface') | (sys_name == 'STO') | (sys_name == 'TiO'):
                n_num = 40
            elif sys_name == 'TiOH':
                n_num = 40
            elif sys_name == 'bulk':
                n_num = 118
            if required_variable == 'dos':
                dataset, z_table = dos_prepare(args.data_num, args.sys_name, args.model_name, n_num, energy_range,
                                               windows, split=split, test=True, save_pth=False)
            elif required_variable == 'dos_split':
                dataset, z_table = dos_prepare(args.data_num, args.sys_name, args.model_name, n_num, energy_range,
                                               windows, split=split, test=True, save_pth=False)
    else:
        if (sys_name.startswith('slab')) | (sys_name == 'surface') | (sys_name == 'STO'):
            n_num = 40
        elif sys_name == 'TiOH':
            n_num = 40
        elif sys_name == 'bulk':
            n_num = 118
        if required_variable == 'dos_split' or required_variable == 'dos':
            energy_range = config["range"]
            windows = config["windows"]
            if sys_name == 'slab':
                path = 'mydata/dos/' + sys_name + f'/data_analysis/{args.model_name}/{required_variable}_2500.pth.tar'
            elif sys_name == 'TiOH':
                path = 'mydata/dos/' + sys_name + f'/data_analysis/{args.model_name}/{required_variable}_2000.pth.tar'
            elif sys_name == 'STO':
                path = 'mydata/dos/' + sys_name + f'/data_analysis/{args.model_name}/{required_variable}_5000.pth.tar'
            z_table = typeread(sys_name)
        #     path = 'mydata/' + required_variable + '/' + sys_name + f'/data_analysis/density.pth.tar'
        #     z_table = typeread(sys_name)
        dataset = Dataset(torch.load(path))
        np.random.seed(seed)
        dataset_indices = np.random.choice(len(dataset), 1697, replace=False)  # 随机选择 data_num 个索引
        # dataset_indices = np.arange(args.data_num)

    if dataset_prepare==False:
        if sys_name == 'slab' and (required_variable == 'dos' or required_variable == 'dos_split'):
            data_num = len(dataset)
            if data_num >= 42:
                dataset_indices = np.arange(data_num - 42)
                last_indices = np.arange(len(dataset) - 42, len(dataset))
                dataset_indices = np.unique(np.concatenate((dataset_indices, last_indices)))
        dataset = [dataset[i] for i in dataset_indices]  # 根据索引选取子集
        # print(len(dataset))

    #写入loss的result文件
    if os.path.exists('../dosnet/data.npz'):
        os.remove('../dosnet/data.npz')

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
        c_save_file = os.path.join('../result/save_model', sys_name, 'c.pth')
        if os.path.exists(c_save_file):
            c_save = torch.load(c_save_file)
            l_dict = c_save['l_dict']
            m_l = sum(list(l_dict.values()))
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
            "correlation": config["correlation"],
            "gate": modules.gate_dict[config["gate"]],
            "interaction_cls_first": modules.interaction_classes["RealAgnosticInteractionBlock"],
            "dropout_rate": config["drop_rate"],
            # "conv_num": config["conv_num"],
            "required_variable": required_variable,
            "windows": windows,
            "m_l":m_l,
            "max_spd":max_spd,
        }
        model = MACE(
            **model_kwargs,
        )
    model.to(device)

    if required_variable=='density' or required_variable=='density_spin':
        # load_model_file = '../result/save_model/' + sys_name.split('_')[0] + '/' + model_name + '/' + required_variable + '.pth'
        load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + required_variable + '.pth'
        model.load_state_dict(torch.load(load_model_file))
        model.eval()
        model_kwargs["test_dataset_size"] = 1
        model_kwargs["density_spacing"] = density_spacing
        if not os.path.exists(
                os.path.join('data_to_be_predicted', 'density', sys_name, 'POSCAR_ALL', str(DATANUM), 'POSCAR')):
            print(f'没有{DATANUM}文件夹里的结构')
        else:
            val_losses = 0
            val_losses_up = 0
            val_losses_mag = 0
            over = 0
            pbar = None
            feature_all = None
            atoms_dict = None
            while True:
                if over == 0:
                    print(f'正在预测{DATANUM}文件夹里的结构')
                a = time.time()
                dataset, over, pbar, typedict, m_l, feature_all, atoms_dict = density_prepare(sys_name=args.sys_name,
                                                                                              density_num=DATANUM,
                                                                                              n_num=n_num, r_cut=r_cut,
                                                                                              save_pth=False, test=True,
                                                                                              patch_num=patch_num,
                                                                                              spin=spin,
                                                                                              max_size=max_size,
                                                                                              POSCAR_only=POSCAR_only,
                                                                                              pbar=pbar, over=over,
                                                                                              feature_filter=feature_filter,
                                                                                              feature_all=feature_all,
                                                                                              atoms_dict=atoms_dict)
                dataset = Dataset(dataset)
                b = time.time()
                print(f'准备数据耗时：{b - a}')
                vi_dict = {}
                vi_dict['feature'] = dataset[0]['feature']
                vi_dict['feature_shape'] = max_size
                vi_dict['num_atoms'] = dataset[0]['num_a']
                vi_dict['d_idx'] = dataset[0]['d_idx']
                vi_dict['copy_line'] = dataset[0]['copy_line']

                model_kwargs["test_dataset"] = dataset

                test_loader = DataLoader(dataset,
                                         batch_size=1,
                                         collate_fn=collate_dicts,
                                         pin_memory=True)

                if POSCAR_only:
                    test_density(model, device, test_loader, DATANUM, POSCAR_only, sys_name, required_variable, vi_dict)
                    c = time.time()
                    print(f'预测耗时：{c - b}')
                    if (over == -1) | (over == 0):
                        print(f'{DATANUM}文件夹里的结构预测完成！')
                        print('+++++++++++++++++++')
                        break
                else:
                    err, err_up, err_mag = test_density(model, device, test_loader, DATANUM, POSCAR_only, sys_name,
                                                        required_variable, vi_dict)
                    val_losses += err
                    if required_variable == 'density_spin':
                        val_losses_mag += err_mag
                        val_losses_up += err_up
                    c = time.time()
                    print(f'预测耗时：{c - b}')
                    print(val_losses)
                    print(val_losses_up)
                    print(val_losses_mag)
                    print(over)
                    with open('结果.txt', 'a') as f:
                        f.write(f'{val_losses}\n')
                    if (over == -1) | (over == 0):
                        print(
                            f'{DATANUM + 1}文件夹里的结构预测完成!预测tot误差：{val_losses};up误差：{val_losses_up};mag误差：{val_losses_mag}')
                        break

    elif (required_variable == 'dos') | (required_variable == 'dos_split'):
        if sys_name == 'TiOH':
            kf = KFold(n_splits=min(10, len(dataset)), shuffle=False)  # 初始化KFold
        else:
            kf = KFold(n_splits=min(10, len(dataset)), shuffle=True, random_state=seed)  # 初始化KFold
        K = -1
        sns_num = len(z_table.keys())
        data_sns_all = torch.zeros([sns_num,sns_num])
        data_sns_scale_all = torch.zeros([sns_num, sns_num])
        data_sns_scaled_all = torch.zeros([sns_num,sns_num])
        data_n_all = torch.zeros([sns_num, sns_num])
        for train_index, val_index in kf.split(dataset):  # 调用split方法切分数据\
            print(val_index)
            K = K + 1
            test_dataset = [dataset[i] for i in val_index]
            test_dataset = dataset
            if dataset_prepare:
                test_dataset = dataset
            if K==1:
                break
            # if K==0:
            #     continue
            test_loader = DataLoader(test_dataset,
                                     batch_size=1,
                                     collate_fn=collate_dicts, )
            load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + required_variable + f'_{K}.pth'
            print(load_model_file)
            # load_model_file = '../result/save_model/' + sys_name + '/' + model_name + '/' + required_variable + f'.pth'
            model.load_state_dict(torch.load(load_model_file))
            model.eval()
            # model_kwargs["test_dataset_size"] = 1
            # model_kwargs["density_spacing"] = density_spacing
            data_sns, data_sns_scale, data_sns_scaled, data_n = test(sys_name, model, device, test_loader, required_variable,
                                                                     energy_range, windows, model_name,sns_num,val_index)
            data_sns_all = data_sns_all + data_sns
            data_sns_scale_all = data_sns_scale_all + data_sns_scale
            data_sns_scaled_all = data_sns_scaled_all + data_sns_scaled
            data_n_all = data_n_all + data_n
        # 创建热力图
        plt.imshow(data_sns_all / data_n_all, cmap='YlGnBu', interpolation='nearest')
        plt.colorbar()  # 添加颜色条
        plt.title('DOS MAE')
        # plt.show()
        plt.close()
        # 创建热力图
        plt.imshow(data_sns_scale_all / data_n_all, cmap='YlGnBu', interpolation='nearest')
        plt.colorbar()  # 添加颜色条
        plt.title('DOS_scaling MAE')
        # plt.show()
        plt.close()
        # 创建热力图
        plt.imshow(data_sns_scaled_all / data_n_all, cmap='YlGnBu', interpolation='nearest')
        plt.colorbar()  # 添加颜色条
        plt.title('DOS_scaled MAE')
        # plt.show()
        plt.close()

        print(f'所有结构预测完成!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train electron density')
    parser.add_argument('--sys_name', type=str)
    parser.add_argument('--required_variable', type=str)
    parser.add_argument('--dataset_prepare', type=str, default='True')
    parser.add_argument('--data_num',type=int)
    parser.add_argument('--model_name', choices=['CNN', 'e3nn', 'mace','Mat2Spec_CNN','Mat2Spec_mace'], help='选择一个选项')
    parser.add_argument('--save_pth', type=str, default='True')

    #计算density的一些参数
    parser.add_argument('--patch_num', type=int,default=4)
    parser.add_argument('--POSCAR_only', type=str, default='False')
    parser.add_argument('--max_size', type=int, default=10000)
    parser.add_argument('--feature_filter',type=str,default='False')

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
    #     # default="32x0e + 32x1o + 32x2e",
    #     default="25x0e + 25x1o + 25x2e",
    # )
    # parser.add_argument('--r_max', help="distance cutoff (in Ang)", type=float, default=5.0)
    # parser.add_argument('--num_interactions', help="number of interactions", type=int, default=2)
    # parser.add_argument(
    #     "--avg_num_neighbors",
    #     help="normalization factor for the message",
    #     type=float,
    #     default=2,
    # )
    # parser.add_argument(
    #     "--correlation", help="correlation order at each layer", type=int, default=3
    # )
    # parser.add_argument(
    #     "--gate",
    #     help="non linearity for last readout",
    #     type=str,
    #     default="tanh",
    #     choices=["silu", "tanh", "abs", "None"],
    # )

    args=parser.parse_args()
    configs = load_config('config.json')
    print(configs[args.required_variable][args.model_name][args.sys_name])
    main(args,configs[args.required_variable][args.model_name][args.sys_name])