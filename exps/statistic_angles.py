import scipy.io as sio
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def main_300W_LP():
    db_name = "300W_LP"
    db_paths = [
        "../300W_LP/AFW", "../300W_LP/AFW_Flip",
        "../300W_LP/HELEN", "../300W_LP/HELEN_Flip",
        "../300W_LP/IBUG", "../300W_LP/IBUG_Flip",
        "../300W_LP/LFPW", "../300W_LP/LFPW_Flip",
        "../300W_LP/AFW", "../300W_LP/AFW_Flip",
        "../AFLW2000"]  # 300W_LP & AFLW2000
    
    total_num = 0
    euler_angles_stat = [[],[],[]]  # pitch, yaw, roll
    
    for db_path in db_paths:    
        onlyfiles_mat = []
        for f in listdir(db_path):
            if isfile(join(db_path, f)) and join(db_path, f).endswith('.mat'):
                onlyfiles_mat.append(f)
        onlyfiles_mat.sort()
        print(db_path, "\t", len(onlyfiles_mat))
        
        for i in tqdm(range(len(onlyfiles_mat))):
            mat_name = onlyfiles_mat[i]
            mat_contents = sio.loadmat(db_path + '/' + mat_name)
            pose_para = mat_contents['Pose_Para'][0]
            pt2d = mat_contents['pt2d']
            
            pitch = pose_para[0] * 180 / np.pi
            yaw = pose_para[1] * 180 / np.pi
            roll = pose_para[2] * 180 / np.pi
             
            if abs(pitch)>99 or abs(yaw)>99 or abs(roll)>99:
                continue
             
            euler_angles_stat[0].append(pitch)
            euler_angles_stat[1].append(yaw)
            euler_angles_stat[2].append(roll)
            total_num += 1

    print("total_num:\t", total_num)
    
    '''Euler Angels Stat'''
    plt.figure(figsize=(10, 5), dpi=100)
    plt.title("300W_LP and AFLW2000")
    interval = 10  # 10 or 15 is better
    bins = 200 // interval
    density = True  # True or False, density=False would make counts
    colors = ['r', 'g', 'b']
    labels = ["Pitch", "Yaw", "Roll"]
    plt.hist(euler_angles_stat, bins=bins, alpha=0.7, density=density, histtype='bar', label=labels, color=colors)
    plt.legend(prop ={'size': 10})
    # plt.xlim(-90, 91)
    plt.xticks(range(-100,101,interval))
    if density: plt.ylabel('Percentage')
    else: plt.ylabel('Counts')
    plt.xlabel('Degree')
    plt.show()
    

def main_BIWI():
    db_path_train = "./BIWI_train.npz"
    db_path_test = "./BIWI_test.npz"
    
    total_num = 0
    euler_angles_stat = [[],[],[]]  # pitch, yaw, roll
    
    for db_path in [db_path_train, db_path_test]:
        db_dict = np.load(db_path)
        print(db_path, list(db_dict.keys()))
        
        for cont_labels in tqdm(db_dict['pose']):
            [yaw, pitch, roll] = cont_labels
            
            if abs(pitch)>90 or abs(yaw)>90 or abs(roll)>90:
                continue

            euler_angles_stat[0].append(pitch)
            euler_angles_stat[1].append(yaw)
            euler_angles_stat[2].append(roll)
            total_num += 1
            
    print("total_num:\t", total_num) 
    
    '''Euler Angels Stat'''
    plt.figure(figsize=(10, 5), dpi=100)
    plt.title("BIWI")
    interval = 10  # 10 or 15 is better
    bins = 180 // interval
    density = True  # True or False, density=False would make counts
    colors = ['r', 'g', 'b']
    labels = ["Pitch", "Yaw", "Roll"]
    plt.hist(euler_angles_stat, bins=bins, alpha=0.7, density=density, histtype='bar', label=labels, color=colors)
    plt.legend(prop ={'size': 10})
    # plt.xlim(-90, 91)
    plt.xticks(range(-90,91,interval))
    if density: plt.ylabel('Percentage')
    else: plt.ylabel('Counts')
    plt.xlabel('Degree')
    plt.show()
    
    
if __name__ == '__main__':
    '''https://github.com/shamangary/FSA-Net'''
    # main_300W_LP()  # total_num 134793
    main_BIWI()  # total_num 15678