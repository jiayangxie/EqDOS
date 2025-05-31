import torch
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt


def pre(x):
    # # 创建原始数据点
    # energy = np.linspace(-15, 15, 500)  # 创建一个包含 10 个元素的等差数列
    # # 创建插值点
    # energy_new = np.linspace(-15, 15, 3000)  # 创建一个包含 100 个元素的等差数列，用于插值
    # x_new = torch.zeros(x.shape[0],3000,x.shape[2])
    # # 进行线性插值
    # #x大小[400,104]->[500,104]
    # for n in range(x.shape[0]):
    #     for j in range(x.shape[2]):
    #         yfit = x[n, :, j]
    #         # dos_fit = interpolate.interp1d(energy, yfit, kind="linear", bounds_error=False, fill_value=0)
    #         # x_new[n, :, j] = torch.from_numpy(dos_fit(energy_new))
    #         x_new[n, :, j] = torch.Tensor(np.interp(energy_new, energy, yfit))
    # y = x_new.permute(2, 0, 1)
    # y = nn.AvgPool1d(6, 6)(y).permute(1,0,2)

    # [batch_size,.permute(1,0,2)windows,104] -> [104,batch_size,windows]->[batch_size,104,windows]
    # y = nn.AvgPool1d(6, 6)(x.permute(2,0,1)).permute(1,0,2)
    # print(torch.sum(torch.sum(y,1),1))
    y = x.permute(0, 2, 1)
    y = torch.cat((y, y), dim=1)
    y[:, :104, 200:] = 0
    y[:, 104:, :200] = 0
    # print(torch.sum(torch.sum(y, 1), 1))
    # print(y.shape)
    return y


# 遍历文件夹中的文件
def get_energy(path):
    info = {}
    with open(path, 'r') as file:
        lines = file.readlines()  # 读取文件的所有行
        for line in lines:
            structure_name, energy = line.strip().split(':')  # 按照 ':' 分割结构名和能量
            info[structure_name] = float(energy)  # 将结构名作为 key，能量作为 value 存储到字典中
    return info


def read_doscar(dir_name, en_range, spd=32, ef=None):
    if ef == None:
        # 读取OUTCAR获取费米能级
        OUT_data_file = os.path.join(dir_name, 'OUTCAR')
        with open(OUT_data_file, 'r') as f:
            outcar_content = f.read()
            f.close()
        # 对于slab，ef默认为None，费米能级使用正则表达式匹配 OUTCAR 中费米能级的信息
        # 对于吸附原子，ef使用holo与lumo的一半
        ef = float(re.findall(r' E-fermi :(.+) \s+XC', outcar_content)[0])
    # 读取DOSCAR文件
    DOS_data_file = os.path.join(dir_name, 'DOSCAR')
    with open(DOS_data_file, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
        dos_n = []
        num = -1
        for d in data[5:]:
            if d.startswith('     20.00000000'):
                num = num + 1
                if num == 1:
                    # 第一个存储的dos是总dos
                    dos_all = dos
                elif num > 1:
                    # 之后出现的dos是每个原子的dos
                    dos_n.append(dos)
                dos = []
            else:
                dos.append([float(x) for x in d.split()])
        # 最终dos_n大小[atom_num,1(energy)+32(dos),windows]
        dos_n.append(dos)
    dos_n = np.array(dos_n).transpose(0, 2, 1)
    # 进行插值计算得到需要的windows大小
    dos = np.zeros((dos_n.shape[0], spd, 3000))
    for i in range(0, dos_n.shape[0]):
        xfit = dos_n[i, 0, :]
        for j in range(1, spd + 1):
            if j > dos_n.shape[1] - 1:
                break
            yfit = dos_n[i, j, :]
            xnew = np.linspace(-en_range, en_range - 0.01, 3000) + ef
            dos[i, j - 1, :] = np.interp(xnew, xfit, yfit)
    dos = torch.nn.AvgPool1d(6, 6)(torch.Tensor(dos)).numpy()
    return dos


def get_xifu_energy(xifu_names):
    xifu_energys = {}
    with open('yuanzi/ef_energy.txt', 'r') as file:
        data = file.read()
        ef_energy = eval(data)
    for xifu_name in xifu_names:
        xifu_energys[xifu_name] = ef_energy[xifu_name]
    return xifu_energys


def get_surface(z, dos, energy_range, windows, xifu_info, xifu_energys, spd=32):
    # 读取吸附原子的DOSCAR，得到大小为[windows,8个轨道]
    # 吸附分子的费米能级是homo，
    dos_spd_save_ml_all = []
    dos_spd_save_vasp_all = []
    energy_values = []
    surs = []
    for (xifu_name, xifu_energy) in xifu_energys.items():
        # ef = xifu_energy[1]
        # dos_yuanzi_spd_new = read_doscar(os.path.join('ad_yuanzi', xifu_name), energy_range, spd=8, ef=ef)[0].transpose(1, 0)
        yuanzi_data = np.load('yuanzi/data_yuanzi.npz')
        dos_yuanzi_spd_new = yuanzi_data['dos'][yuanzi_data['xifu_names'] == xifu_names][0]
        # 检查一下这个结构在数据集中有没有对应吸附能数据，如果没有就不进行预测跳过。
        surface = {key: value for key, value in xifu_info.items() if key.startswith(z + '_' + xifu_name + '_')}
        if len(surface) == 0:
            return None, None, None
        else:
            for (sur, energy_value) in surface.items():
                # 获取吸附位点信息
                weidian = [int(x) for x in sur.split('_')[-1].split(",")]
                weidian_num = len(weidian)
                dos_spd_save_ml = dos[0][weidian].transpose(1, 0, 2).reshape(windows, spd * weidian_num)
                dos_spd_save_vasp = dos[1][weidian].transpose(1, 0, 2).reshape(windows, spd * weidian_num)
                # 对不足三个原子的地方进行补零操作
                dos_spd_save_ml = np.pad(dos_spd_save_ml, ((0, 0), (0, 32 * (3 - weidian_num)))) / weidian_num
                dos_spd_save_vasp = np.pad(dos_spd_save_vasp, ((0, 0), (0, 32 * (3 - weidian_num)))) / weidian_num
                # 最后补上吸附分子的dos->[windows,104]
                dos_spd_save_ml = np.concatenate((dos_spd_save_ml, dos_yuanzi_spd_new), 1)
                dos_spd_save_vasp = np.concatenate((dos_spd_save_vasp, dos_yuanzi_spd_new), 1)
                # print(sur)
                # print(np.sum(dos_spd_save_vasp[:,0:32]))
                # print(np.sum(dos_spd_save_vasp[:, 32:64]))
                # print(np.sum(dos_spd_save_vasp[:, 64:96]))
                # print(np.sum(dos_spd_save_vasp[:, 96:]))
                # xnew = np.linspace(-15, 15, 500)
                # for i in range(0, dos[1].shape[0]):
                #     print(dos.shape)
                #     plt.plot(xnew, np.sum(dos[1][i], 1)-np.sum(dos[1][i+1], 1))
                #     print(np.sum(np.sum(dos[1][i], 1),0)/3)
                #     plt.show()
                #     plt.close()
                # print('++++++++++++++++++++++')

                # xnew = np.linspace(-15, 15, 500)
                # for i in range(96,dos_spd_save_vasp.shape[1]):
                #     plt.plot(xnew, dos_spd_save_vasp[:, i])
                #     plt.plot(xnew,dos[1][weidian][2][:,i-64]/weidian_num)
                #     plt.show()
                #     plt.close()

                # 将不同吸附位点的存储到同一数组
                dos_spd_save_ml_all.append(dos_spd_save_ml)
                dos_spd_save_vasp_all.append(dos_spd_save_vasp)
                energy_values.append(energy_value)
                surs.append(sur)
    # 将ml与vasp的数据进行合并
    dos_spd_save_ml_all = torch.Tensor(dos_spd_save_ml_all).unsqueeze(1)
    dos_spd_save_vasp_all = torch.Tensor(dos_spd_save_vasp_all).unsqueeze(1)
    # print(surs)
    # print(torch.sum(torch.sum(dos_spd_save_vasp_all.squeeze(1),1),1))
    # print('+++++++++++++')
    dos_save = torch.cat((dos_spd_save_ml_all, dos_spd_save_vasp_all), 1)
    return dos_save, torch.Tensor(energy_values), surs


def mape_loss_func(preds, labels):
    return torch.abs((labels - preds) / labels).mean()


# 绘制结果图
def draw_result(xifu_name, ml_xifu, ml_all, vasp, zs, name):
    xifu_color = {'C': (230 / 255, 85 / 255, 13 / 255),
                  'H': (49 / 255, 130 / 255, 189 / 255),
                  'O': (117 / 255, 107 / 255, 177 / 255),
                  'N': (49 / 255, 163 / 255, 84 / 255),
                  'S': (99 / 255, 99 / 255, 99 / 255),
                  'CH': (71 / 255, 141 / 255, 205 / 255),
                  'HO': (71 / 255, 141 / 255, 205 / 255),
                  'HS': (71 / 255, 141 / 255, 205 / 255),
                  'CH2': (71 / 255, 141 / 255, 205 / 255),
                  'CH3': (71 / 255, 141 / 255, 205 / 255),
                  'H2O': (71 / 255, 141 / 255, 205 / 255),
                  'all': ((255 / 255, 255 / 255, 255 / 255))}
    mae_xifu = torch.nn.functional.l1_loss(ml_xifu, vasp)
    mape_xifu = mape_loss_func(ml_xifu, vasp) * 100
    mae_all = torch.nn.functional.l1_loss(ml_all, vasp)
    mape_all = mape_loss_func(ml_all, vasp) * 100
    plt_range = math.floor(torch.min(vasp)) - 1
    # mae = torch.nn.functional.l1_loss(ml,vasp)
    # print(f'{name}的mae是：{mae}\n{name}的mape是：{mape}')
    print(
        f'{xifu_name}|{ml_all.shape[0]}|{torch.mean(vasp):4f}|{mae_xifu:4f}|{mape_xifu:4f}|{mae_all:4f}|{mape_all:4f}')

    # 画图要画 结构->energy 与 vasp计算energy 对比
    # 计算相关系数
    corr_coef = torch.nn.functional.cosine_similarity(ml_all, vasp, dim=0)
    ml_all = ml_all.cpu().detach().numpy()
    vasp = vasp.cpu().detach().numpy()
    plt.scatter(ml_all, vasp, color=xifu_color[xifu_name], alpha=1, s=0.5)
    # 添加 y = x 的辅助线
    plt.plot([plt_range, 0], [plt_range, 0], color='green', linestyle='--')
    plt.xlabel('ML')
    plt.ylabel('VASP')
    plt.title('Parity Plot of out')
    plt.grid(False)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.text(0.45, 0.75, f'MAE={mae_all:.4f}\nR^2={corr_coef:.4f}',
             transform=ax.transAxes, horizontalalignment='right')
    save_dir = f'result/{xifu_name}'
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, f'{name}.jpg'))
    plt.close()


if __name__ == '__main__':
    # data.npz N * 3000 * 104
    # range(-15eV, 15eV, 0.01eV)=3000
    # (s*2 + p*6 + d*10 + f*14) * 3 + (s*2 + p*6)=104
    s_name = ''
    xifu_names_all = [['HS'],['CH2'],['CH3'],['H2O']]
    # xifu_names_all = [['C'], ['H'], ['O'], ['N'], ['S'], ['CH'], ['HO']]
    # 加载模型
    dosnet = torch.jit.load('model.pt').cuda().eval()
    # dosnet = torch.load('./dosnet.pkl').cuda().eval()
    surs_new = []

    # for (xifu_i,xifu_names) in enumerate(xifu_names_all):
    #     # print(xifu_names[0])
    #
    #     # metadata = np.load('test/data.npz')['meta']
    #     # i_all = []
    #     # for (i,meta) in enumerate(metadata):
    #     #     if meta[0] in xifu_names and meta[1].startswith(s_name) and meta[2]=='top':
    #     #         i_all.append(i)
    #     # metadata = metadata[i_all]
    #     # source = np.load('test/data.npz')['dos'][i_all]
    #     # source = pre(torch.nn.AvgPool1d(6, 6)(torch.Tensor(source).permute(0,2,1)).permute(0,2,1)).numpy()
    #     # target = np.load('test/data.npz')['target'][i_all]
    #
    #     xifu_info = get_energy('./energy_xifu.txt')
    #     data = np.load('./data.npz')
    #     dataset = data['dos'].astype('float32')
    #     structure_name = data['z']
    #     surs = []
    #     i = 0
    #
    #     # 读取所有需要吸附质的费米能级
    #     xifu_energys = get_xifu_energy(xifu_names)
    #
    #     for z, dos in zip(structure_name, dataset):
    #         if z.startswith(s_name):
    #             surface_dos, energy_value, sur = get_surface(z, dos, 15, 500, xifu_info, xifu_energys)
    #             if isinstance(energy_value, torch.Tensor):
    #                 if i == 0:
    #                     surface_doses = surface_dos
    #                     energy_values = energy_value
    #                 else:
    #                     surface_doses = torch.cat((surface_doses, surface_dos), 0)
    #                     energy_values = torch.cat((energy_values, energy_value), 0)
    #                 surs = surs + sur
    #                 i += 1
    #
    #     dataset = surface_doses
    #     xifu_energy = energy_values
    #     # print('已将所有结构处理完成！')
    #     # # 根据dataset中的'z'索引提取 xifu_info 中的能量信息
    #     # xifu_energy = torch.Tensor([xifu_info.get(key, 0) for key in structure_name])
    #
    #     # dosnet = torch.load('./dosnet.pkl').cuda().eval()
    #     # dos = pre(torch.from_numpy(dataset)).cuda()
    #     # out = dosnet(dos)
    #     # for i in range(out.shape[0]):
    #     #     print(f'{out[i]}|{xifu_energy[i]}')
    #
    #     # print(f'第i个元素|用ml计算出dos预测|用vasp计算出dos预测|用vasp直接计算的吸附能|dos预测模型造成的误差|dos-吸附能总误差')
    #
    #     dos_ml = pre(dataset[:, 0, :, :]).cuda()
    #     dos_vasp = pre(dataset[:, 1, :, :]).cuda()
    #     xifu_energy = xifu_energy.cuda()
    #     out_ml = torch.zeros(xifu_energy.shape).cuda()
    #     out_vasp = torch.zeros(xifu_energy.shape).cuda()
    #
    #     # print(torch.sum(dos_vasp[0,:,:][:96],1)+torch.sum(dos_vasp[0,:,:][104:200],1)-torch.Tensor(np.sum(source[1,:,:][:96],1)-np.sum(source[1,:,:][104:200],1)).cuda())
    #     # # print(np.sum(source[1,:,:][:96],1)+np.sum(source[1,:,:][104:200],1))
    #     # print(torch.sum(dos_vasp[0,:,:][96:104],1)+torch.sum(dos_vasp[0,:,:][200:],1)-torch.Tensor(np.sum(source[1,:,:][96:104],1)-np.sum(source[1,:,:][200:],1)).cuda())
    #
    #     # xnew = np.linspace(-15, 14.99, 500)
    #     # for i in range(8):
    #     #     # plt.plot(xnew, source[1, :, :][96+i] + source[1, :, :][200+i])
    #     #     plt.plot(xnew, source[1, :, :][96+i] + source[1, :, :][200+i] -
    #     #                     dos_vasp.cpu().numpy()[0, :, :][96+i] - dos_vasp.cpu().numpy()[0, :, :][200+i])
    #     #     plt.show()
    #     #     plt.close()
    #     # # print(np.sum(source[1,:,:][96:104],1)+np.sum(source[1,:,:][200:],1))
    #     # plt.plot(xnew, np.sum(source[1,:,:][96:104], 0) + np.sum(source[1,:,:][200:], 0))
    #     # plt.plot(xnew, np.sum(source[1,:,:][96:104],0)+np.sum(source[1,:,:][200:],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][96:104],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][200:],0))
    #     # plt.show()
    #     # plt.close()
    #     # plt.plot(xnew, np.sum(source[1,:,:][:96], 0) + np.sum(source[1,:,:][104:200], 0))
    #     # plt.plot(xnew, np.sum(source[1,:,:][:96],0)+np.sum(source[1,:,:][104:200],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][:96],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][104:200],0))
    #     # plt.show()
    #     # plt.close()
    #
    #
    #     with torch.no_grad():
    #         for n in range(dataset.shape[0]):
    #             out_ml[n] = dosnet(dos_ml[n].unsqueeze(0))
    #             out_vasp[n] = dosnet(dos_vasp[n].unsqueeze(0))
    #
    #     # draw_result(xifu_names[0],out_vasp,out_ml, xifu_energy[:], surs,name=xifu_names[0])
    #     if xifu_i==0:
    #         out_vasp_all = out_vasp
    #         out_ml_all = out_ml
    #         xifu_energy_all = xifu_energy
    #     else:
    #         out_vasp_all = torch.cat((out_vasp_all,out_vasp),0)
    #         out_ml_all = torch.cat((out_ml_all, out_ml), 0)
    #         xifu_energy_all = torch.cat((xifu_energy_all,xifu_energy),0)

    # draw_result('all', out_vasp_all, out_ml_all, xifu_energy_all, surs,name='all_result')

    # 开始画全部的图

    for (xifu_i, xifu_names) in enumerate(xifu_names_all):
        xifu_color = {'C': (230 / 255, 85 / 255, 13 / 255),
                      'H': (49 / 255, 130 / 255, 189 / 255),
                      'O': (117 / 255, 107 / 255, 177 / 255),
                      'N': (49 / 255, 163 / 255, 84 / 255),
                      'S': (99 / 255, 99 / 255, 99 / 255),
                      'CH': (71 / 255, 141 / 255, 205 / 255),
                      'HO': (71 / 255, 141 / 255, 205 / 255),
                      'HS': (71 / 255, 141 / 255, 205 / 255),
                      'CH2': (71 / 255, 141 / 255, 205 / 255),
                      'CH3': (71 / 255, 141 / 255, 205 / 255),
                      'H2O': (71 / 255, 141 / 255, 205 / 255), }
        xifu_info = get_energy('./energy_xifu.txt')
        data = np.load('../result/dos/data.npz')
        dataset = data['dos'].astype('float32')
        structure_name = data['z']
        structure_name = structure_name[:len(structure_name)]
        dataset = dataset[:len(dataset)]
        surs = []
        i = 0

        # 读取所有需要吸附质的费米能级
        xifu_energys = get_xifu_energy(xifu_names)
        for z, dos in zip(structure_name, dataset):
            if z.startswith(s_name):
                surface_dos, energy_value, sur = get_surface(z, dos, 15, 500, xifu_info, xifu_energys)
                if isinstance(energy_value, torch.Tensor):

                    if i == 0:
                        surface_doses = surface_dos
                        energy_values = energy_value
                    else:
                        surface_doses = torch.cat((surface_doses, surface_dos), 0)
                        energy_values = torch.cat((energy_values, energy_value), 0)
                    surs = surs + sur
                    i += 1

        dataset = surface_doses[:dataset.shape[0]//5]
        xifu_energy = energy_values[:dataset.shape[0]//5]
        print(dataset.shape[0])

        dos_ml = pre(dataset[:, 0, :, :]).cuda()
        dos_vasp = pre(dataset[:, 1, :, :]).cuda()
        xifu_energy = xifu_energy.cuda()
        out_ml = torch.zeros(xifu_energy.shape).cuda()
        out_vasp = torch.zeros(xifu_energy.shape).cuda()

        with torch.no_grad():
            for n in range(dataset.shape[0]):
                out_ml[n] = dosnet(dos_ml[n].unsqueeze(0))
                out_vasp[n] = dosnet(dos_vasp[n].unsqueeze(0))

        print(dataset.shape[0])
        # 数据啊啊啊啊啊
        # indices = torch.where(torch.abs(out_ml - xifu_energy[:]) <= 0.75)[0]
        # print(surs)
        # surs_new = [surs[i] for i in indices.tolist()]
        # xifu_energy_new = xifu_energy[indices]

        # ## 打开一个文件用于写入，如果文件不存在则创建
        # with open('output.txt', 'w') as file:
        #     # 遍历两个列表的索引
        #     for i in range(len(surs_new)):
        #         # 将每个元素按照指定格式写入文件
        #         file.write(f"{surs_new[i]}: {xifu_energy_new[i]}\n")
        # 到此结束

        mae_xifu = torch.nn.functional.l1_loss(out_vasp, xifu_energy[:])
        mape_xifu = mape_loss_func(out_vasp, xifu_energy[:]) * 100
        mae_all = torch.nn.functional.l1_loss(out_ml, xifu_energy[:])
        mape_all = mape_loss_func(out_ml, xifu_energy[:]) * 100
        plt_range = -10
        corr_coef = torch.nn.functional.cosine_similarity(out_ml, xifu_energy[:], dim=0)
        out_ml = out_ml.cpu().detach().numpy()
        # xifu_energy[:] = xifu_energy[:]
        plt.scatter(out_ml, xifu_energy[:].cpu().detach().numpy(), color=xifu_color[xifu_names[0]], alpha=1, s=0.5)
        # 添加 y = x 的辅助线
        plt.plot([plt_range, 0], [plt_range, 0], color='grey', alpha=0.3, linestyle='--')
        plt.xlabel('ML')
        plt.ylabel('VASP')
        plt.title('Parity Plot of out')
        plt.grid(False)
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.text(0.25, 0.75, f'MAE={mae_all:.4f}\nR^2={corr_coef:.4f}',
        #          transform=ax.transAxes, horizontalalignment='right')

    save_dir = f'result/all'
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, f'all.jpg'))