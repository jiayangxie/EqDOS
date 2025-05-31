import os
from tqdm import tqdm
import argparse

from .poscar_deal import poscar_read,atom_deal
from .dos_deal import dos_deal
from ..tools.Dataset import Dataset



def dos_get(i, dir_name, n_num,typedict,split,model_name,en_range,windows,spd_dict,max_spd):
    if os.path.exists(os.path.join(dir_name, f"{i}.vasp")):
        POS_data_file =  os.path.join(dir_name, f"{i}.vasp")
    elif os.path.exists(os.path.join(dir_name,str(i),'POSCAR')):
        POS_data_file = os.path.join(dir_name,str(i),'POSCAR')
    atoms, xyz, n, xyz_num, abc = poscar_read(POS_data_file)
    weighted_onehot, edge_num, edge_index, edge_veg, shifts, unit_shifts, (atom_conb, type),typedict = atom_deal(n,abc,xyz,atoms,model_name, typedict, n_num)

    dos,spd,l_shape, scaling = dos_deal(dir_name,i,xyz_num,en_range,windows,split,spd_dict,max_spd,n)
    if dos is None:
        return None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,typedict

    num_atom = [int(len(n))]
    # return nxyz, xyz,abc, weighted_onehot, n, num_atom,dos,dos_read.shape[1],scaling, edge_num,edge_index,edge_length.unsqueeze(1),edge_veg,(atom_conb, type),typedict
    return n, xyz, abc, weighted_onehot, num_atom,dos , l_shape, scaling, edge_num,edge_index,edge_veg,shifts,unit_shifts,spd,(atom_conb, type), typedict

def typeread(sys_name):
    with open(os.path.join('../result/save_model/', sys_name.split('_')[0], 'typedict.txt'), "r", encoding="utf-8") as f:
        dataline = f.read().splitlines()
    typedict={}
    for data in dataline:
        typedict[float(data.split(' ')[0])]=int(data.split(' ')[1])
    return typedict

def spd_dict_get(sys_name):
    # 获得一个列表，key是原子序号，value是哪个轨道上有值（用1表示），哪个轨道上没有值（用0表示）
    spd_dict = {}
    with open(f'../result/save_model/{sys_name}/spd_dict.txt', 'r') as f:
        # 从文件中读取内容
        contents = f.readlines()
        f.close()
    for line in contents:
        parts = line.strip().split(':')
        key = float(parts[0])
        # 注意这里需要将字符串形式的列表转换回原始列表
        value = list(map(int, parts[1][1:-1].split(' ')))
        spd_dict[key] = value
    return spd_dict

def dos_prepare(density_num,sys_name,model_name,n_num,en_range,windows,save_pth=True,test=False,split=False,train_data_num=None):
    if test:
        dir_name = 'data_to_be_predicted/dos/'+sys_name + '/dos_all/'
    else:
        dir_name = 'mydata/dos/'+sys_name + '/dos_all/'
    if os.path.exists(dir_name):
        pass
    else:
        import tarfile
        tar_file_path = f'mydata/dos/{sys_name}/{sys_name}_DOS_dataset.tar.gz'
        if os.path.exists(tar_file_path):
            print('文件尚未解压，正在解压缩')
            with tarfile.open(tar_file_path, 'r:gz') as tar:
                tar.extractall(f'{sys_name}/')  # 解压缩全部内容到当前目录
            os.rename(f'mydata/dos/{sys_name}/{sys_name}_DOS_dataset/',dir_name)
            print('解压缩成功！')
        else:
            print('没有文件')
    m = 0
    n = 0
    if density_num==None:
        import glob
        file_pattern = '[0-9]*'  # 匹配由至少一个数字组成的文件名
        # 根据文件名排序并获取文件列表
        file_list = sorted(glob.glob(os.path.join(dir_name, file_pattern)))  # 获取所有文件
        if file_list:
            base_name = os.path.basename(file_list[-1])  # 获取文件名部分
            density_num = int(os.path.splitext(base_name)[0])  # 去除文件扩展名
        else:
            import tarfile
            tar_file_path = f'{sys_name}/{sys_name}_DOS_dataset.tar.gz'
            if os.path.exists(tar_file_path):
                print('文件尚未解压，正在解压缩')
                with tarfile.open(tar_file_path, 'r:gz') as tar:
                    tar.extractall(f'mydata/dos/{sys_name}/')  # 解压缩全部内容到当前目录
                os.rename(f'mydata/dos/{sys_name}/{sys_name}_DOS_dataset/', dir_name)
                print('解压缩成功！')
            else:
                print('没有文件')
    # nxyzs = [[] for i in range(density_num)]
    ns = [[] for i in range(density_num)]
    xyzs = [[] for i in range(density_num)]
    abc = [[] for i in range(density_num)]
    weighted_onehots = [[] for i in range(density_num)]
    num_a = [[] for i in range(density_num)]
    dos = [[] for i in range(density_num)]
    l_shape = [[] for i in range(density_num)]
    scaling = [[] for i in range(density_num)]
    edge_num = [[] for i in range(density_num)]
    edge_index = [[] for i in range(density_num)]
    edge_vec = [[] for i in range(density_num)]
    shifts = [[] for i in range(density_num)]
    unit_shifts = [[] for i in range(density_num)]
    spd = [[] for i in range(density_num)]
    pbar = tqdm(total=density_num)
    cell_and_atom_type_dict = {}



    if split:
        max_spd = 0
        spd_dict = spd_dict_get(sys_name)
        for v in spd_dict.values():
            count_ones = v.count(1)  # 计算当前列表中 1 的数量
            if count_ones > max_spd:
                max_spd = count_ones  # 更新最大值
    else:
        max_spd = 32
        spd_dict = None

    if test:
        typedict = typeread(sys_name)
    else:
        typedict = {}
    while True:
        if os.path.exists(f'{dir_name+str(m)}.npy') or os.path.exists(os.path.join(dir_name,str(m),'DOSCAR')):
            ns[n],xyzs[n], abc[n],weighted_onehots[n], num_a[n], dos[n] ,l_shape[n],scaling[n], edge_num[n],edge_index[n],edge_vec[n],shifts[n],unit_shifts[n],spd[n],cell_and_atom,typedict= dos_get(m,dir_name,n_num,typedict,split,model_name,en_range,windows,spd_dict,max_spd)
            cell_and_atom_type_dict[cell_and_atom] = cell_and_atom_type_dict.get(cell_and_atom, 0) + 1
            m = m + 1
            if ns[n] != None:
                if n == density_num - 1:
                    break
                if dos[n] is None:
                    '结构异常,不参加训练'
                else:
                    n = n + 1
                    pbar.update(1)
            else:
                continue
        else:
            del ns[n - density_num:]
            del xyzs[n - density_num:]
            del abc[n - density_num:]
            del weighted_onehots[n - density_num:]
            del num_a[n - density_num]
            del dos[n-density_num:]
            del l_shape[n - density_num:]
            del scaling[n-density_num:]
            del edge_index[n-density_num:]
            # del edge_length[j-density_num:]
            del edge_vec[n-density_num:]
            del shifts[n-density_num:]
            del unit_shifts[n-density_num:]
            del edge_num[n-density_num:]
            del spd[n-density_num:]
            print(f'预计包装{density_num}bulk个结构，但文件夹里只有{n}bulk个结构')
            density_num = n
            break
    if test:
        pass
    else:
        typedict_save_file = os.path.join('../result/save_model/', sys_name, 'typedict.txt')
        if os.path.exists(typedict_save_file):
            os.remove(typedict_save_file)
        with open(typedict_save_file, 'a') as f:
            for key, value in typedict.items():
                f.write(f'{key} {value}\n')
    pbar.close()
    props = {
        ##s输入数据
        'z': ns,
        'pos': xyzs,
        'abc': abc,
        'x': weighted_onehots,
        'num_a':num_a,
        'dos':dos,
        # 'l_shape':l_shape,
        'scaling':scaling,
        'edge_index': edge_index,
        # 'edge_length': edge_length,
        'edge_vec': edge_vec,
        'shift':shifts,
        'unit_shift':unit_shifts,
        'edge_num': edge_num,
        'spd': spd,
    }
    dataset = Dataset(props)
    if save_pth==True:
        if split:
            path = 'mydata/dos/' + sys_name + f'/data_analysis/{model_name}/dos_split_{train_data_num}.pth.tar'
        else:
            path = 'mydata/dos/' + sys_name + f'/data_analysis/{model_name}/dos_{train_data_num}.pth.tar'
        if os.path.exists('mydata/dos/'+sys_name+'/data_analysis/'+model_name) == False:
            os.mkdir('mydata/dos/'+sys_name+'/data_analysis/'+model_name)
        with open('mydata/dos/'+sys_name+'/data_analysis/'+model_name+ f'/traindata_analysis_{train_data_num}.txt', "w+",
                  encoding="utf-8") as f:
            f.write(f"目前数据集用到的结构总共有：{density_num}\n")
            f.write(f"总类分别为：\n")
            for atom_conb, value in cell_and_atom_type_dict.items():
                f.write(f'{(atom_conb)}:{value}\n')
    if save_pth==True:
        if split==True:
            dataset.save(path)
        else:
            dataset.save(path)
        return dataset,typedict
    else:
        return dataset,typedict


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train')
    # parser.add_argument("--multi_adsorbate",default=0,type=int,help="train for single adsorbate (0) or multiple (1) (default: 0)")
    parser.add_argument('--sys_name', type=str)
    parser.add_argument('--density_num',type=int,default=None)
    parser.add_argument('--model_name',type=int,default=None)
    args = parser.parse_args()
    dos_prepare(args.density_num,args.sys_name,args.model_name)

