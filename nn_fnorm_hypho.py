import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from math import log10, floor
import plotly.graph_objects as go

def plot(src,target,title,idx):
    #combined plot
    n=1000
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,10)
    plt.scatter(src[:n, 0], src[:n, 1],color='r',label='Src_class_0',marker='o')
    plt.scatter(src[n:, 0], src[n:, 1],color='r',label='Src_class_1',marker='x')
    plt.scatter(target[:n, 0], target[:n, 1],color='g',label='Target_class_0',marker='o')
    plt.scatter(target[n:, 0], target[n:, 1],color='g',label='Target_class_1',marker='x')
    plt.title(title)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    # idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+title
    plt.savefig(f_name+'.png')
#     plt.axis('scaled')
    # plt.show()
    
def pca(samp):
    pca = PCA(n_components=2)
    pca.fit(samp)
    new_df=pca.transform(samp)
    # new_df = np.array(new_df)
    return new_df,pca.components_.T,pca.explained_variance_


def inv_sigmoid(value):
    return np.log(value/(1-value))

def gaussian_mixture_model(mus,covs,n):
    # mus = [np.array([-2, 2]), np.array([2, -2])]
    # covs = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
    pis = []
    if len(mus)==2:
        pis = np.array([0.5,0.5])
    else:
        pis = np.array([0.34, 0.33,0.33])
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
    assert np.isclose(acc_pis[-1], 1)

    # n = 1000
    samples = []

    for i in range(n):
        # sample uniform
        r = np.random.uniform(0, 1)
        # select gaussian
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break

        selected_mu = mus[k]
        selected_cov = covs[k]

        # sample from selected gaussian
        lambda_, gamma_ = np.linalg.eig(selected_cov)

        dimensions = len(lambda_)
        # sampling from normal distribution
        y_s = np.random.uniform(0, 1, size=(dimensions * 1, 3))
        x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
        # transforming into multivariate distribution
        x_multi = (x_normal * lambda_) @ gamma_ + selected_mu
        samples.append(x_multi.tolist()[0])

    samples = np.array(samples)
    return samples

def affine_transformation(rot_mat,df):
    df_rot=np.dot(df,rot_mat)
    return df_rot

def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))

def rot_mat(deg):
    # deg=deg
    rad_rot=np.deg2rad(deg)
    homography_mat = np.array([[np.cos(rad_rot),np.sin(rad_rot)],[-np.sin(rad_rot),np.cos(rad_rot)]]) #counterclockwise positive
    return homography_mat

def mse(data,proj_points):
    error = ((data - proj_points)**2).mean(axis=None)
    return error

if __name__=='__main__':
    idx=datetime.now()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    
    generate_data=False
    save_flag=False

    homography_mat=np.ones((2,2))
    rotation_flag=False
    scale_flag=True
    if rotation_flag:
        deg=60
        rad_rot=np.deg2rad(deg)
        homography_mat = np.array([[np.cos(rad_rot),np.sin(rad_rot)],[-np.sin(rad_rot),np.cos(rad_rot)]]) #counterclockwise positive
    elif scale_flag:
        scale_x = 2
        scale_y = 0.5
        homography_mat = np.array([[scale_x,0],[0,scale_y]])
    else:
        homography_mat=np.round(np.random.uniform(-1,1,(2,2)),decimals=5) #2x2 #src x T = target


    if (generate_data):
        # Generate data
        n=1000
        # source domain: GMM - class 0
        src_mus_c0 = [np.array([-2, 8]), np.array([-2,-6]),np.array([-5,3])]
        src_covs_c0 = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.2], [-0.2, 1]]),np.array([[0.5, -0.3], [-0.3, 1]])]
        src_c0 = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)

        # source domain: GMM - class 1
        src_mus_c1 = [np.array([10, 0]), np.array([-15, 5])]
        src_covs_c1 = [np.array([[0.8, 0.8], [0.8, 0.2]]), np.array([[1, -0.8], [-0.8, 1.5]])]
        src_c1 = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
        
        # target domain: GMM - class 1 (rotated source class 1)
        target_c1_before = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
        target_c1 = affine_transformation(homography_mat,target_c1_before)

        # target domain: GMM - class 0 (rotated target class 0)
        target_c0_before = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)
        target_c0 = affine_transformation(homography_mat,target_c0_before)

        # combined dataset - source
        src_data = np.concatenate((src_c0,src_c1),axis=0)
        # combined dataset - target
        target_data = np.concatenate((target_c0,target_c1),axis=0)
    else:
        src_data=np.load('Src_scale.npy')
        target_data=np.load('Target_scale.npy')
    
    plot(src_data,target_data,'Sampled Data',idx)
    
    # Save dataset (optional)
    
    if (save_flag):
        np.save("Src_scale.npy",src_data)
        np.save("Target_scale.npy",target_data)

    pca_space_flag = False
    if (pca_space_flag):
        # pca on source : X_pca_src
        X_pca_src,X_s_ori,_ = pca(src_data) #X_s: Dxd (column vector)

        # pca on target : X_pca_target
        X_pca_target,X_t_ori,_ = pca(target_data) #X_t: Dxd (column vector)

        plot(X_pca_src,X_pca_target,'PCA_space',idx)
    else:
        X_pca_src = src_data
        X_pca_target = target_data

    # cov of X_pca_target : Cov_target_pca
    Cov_target_pca=np.cov(X_pca_target.T)

    rot_angles=[i for i in range(0,365,5)]

    f_norm_arr=[]
    mse_arr=[]

    if rotation_flag:
        src_rot = affine_transformation(rot_mat(60),src_data)
        
        for x_deg in rot_angles:
            rotation_mat=rot_mat(x_deg)
            src_aligned_xDeg = affine_transformation(rotation_mat,X_pca_src)
            # check Forbenius norm
            # cov of src_aligned : Cov_src_pca_aligned
            Cov_src_pca_aligned_xDeg = np.cov(src_aligned_xDeg.T)
            forb_norm = np.sum(np.square(Cov_src_pca_aligned_xDeg-Cov_target_pca))
            f_norm_arr.append(forb_norm)
            if not pca_space_flag:
                mse_error=mse(src_rot,src_aligned_xDeg)
                mse_arr.append(mse_error)

        # plot
        plt.clf()
        plt.scatter(rot_angles,mse_arr,color='r',label='MSE loss')
        plt.scatter(rot_angles,f_norm_arr,color='b',label='Forbenius norm')
        plt.title("Loss curve")
        plt.ylabel('Loss')
        plt.xlabel('Rotation angles')
        plt.legend()
        dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+'loss curve'
        plt.savefig(f_name+'.png')

        # plot
        plt.clf()
        plt.scatter(rot_angles,mse_arr,color='r',label='MSE loss')
        plt.title("MSE Loss curve")
        plt.ylabel('Loss')
        plt.xlabel('Rotation angles')
        plt.legend()
        dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+'mse loss curve'
        plt.savefig(f_name+'.png')

        # plot
        plt.clf()
        plt.scatter(rot_angles,f_norm_arr,color='b',label='Forbenius norm')
        plt.title("Forbenius Loss curve")
        plt.ylabel('Loss')
        plt.xlabel('Rotation angles')
        plt.legend()
        dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        f_name=dt_string+'fnorm loss curve'
        plt.savefig(f_name+'.png')
    
    scale_x_val=np.arange(0,3.5,0.5)
    scale_y_val=np.arange(0,3.5,0.5)
    
    scale_xy = [] #[[scale_x1,scale_y1]]
    for i in range(len(scale_x_val)):
        for j in range(len(scale_y_val)):
            scale_xy.append([scale_x_val[i],scale_y_val[j]])
    scale_xy = np.array(scale_xy)
    if scale_flag:
        src_scale = affine_transformation(np.array([[scale_x,0],[0,scale_y]]),src_data)
        for sc_val in scale_xy:
            scale_mat=np.array([[sc_val[0],0],[0,sc_val[1]]])
            src_aligned_xScale = affine_transformation(scale_mat,X_pca_src)
            # check Forbenius norm
            # cov of src_aligned : Cov_src_pca_aligned
            Cov_src_pca_aligned_xScale = np.cov(src_aligned_xScale.T)
            forb_norm = np.sum(np.square(Cov_src_pca_aligned_xScale-Cov_target_pca))
            f_norm_arr.append(forb_norm)
            if not pca_space_flag:
                mse_error=mse(src_scale,src_aligned_xScale)
                mse_arr.append(mse_error)

        # plot //changes
        # plt.clf()
        # fig = go.Figure(data=[go.Scatter3d(x=scale_xy[:,0], y=scale_xy[:,1], z=mse_arr,
        #                            mode='markers')])
        # fig = go.Figure(data=[go.Scatter3d(x=scale_xy[:,0], y=scale_xy[:,1], z=f_norm_arr,
        #                            mode='markers')])    
        # fig.show()
        # plt.clf()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(scale_xy[:,0], scale_xy[:,1], mse_arr, c=mse_arr, cmap='Greens',label='MSE Loss')
        # ax.scatter(scale_xy[:,0], scale_xy[:,1], f_norm_arr,c=f_norm_arr, cmap='Reds', label='Forbenius norm')
        # ax.set_xlabel('x_scale')
        # ax.set_ylabel('y_scale')
        # ax.set_zlabel('Loss')
        
        # plt.title("Loss curve")
        # plt.legend()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        # f_name=dt_string+'loss curve'
        # plt.savefig(f_name+'.png')
        

        # plot
        plt.clf()
        fig = go.Figure(data=[go.Scatter3d(x=scale_xy[:,0], y=scale_xy[:,1], z=mse_arr,
                                   mode='markers')])    
        fig.show()
        # plt.clf()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(scale_xy[:,0], scale_xy[:,1], mse_arr, c=mse_arr,cmap='Greens',label='MSE Loss')
        # ax.set_xlabel('x_scale')
        # ax.set_ylabel('y_scale')
        # ax.set_zlabel('Loss')
        
        # plt.title("MSE loss curve")
        # plt.legend()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        # f_name=dt_string+'mse loss curve'
        # plt.savefig(f_name+'.png')


        # plot
        plt.clf()
        fig = go.Figure(data=[go.Scatter3d(x=scale_xy[:,0], y=scale_xy[:,1], z=f_norm_arr,
                                   mode='markers')])    
        fig.show()
        # plt.clf()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(scale_xy[:,0], scale_xy[:,1], f_norm_arr, c=f_norm_arr,cmap='Reds',label='Forbenius norm')
        # ax.set_xlabel('x_scale')
        # ax.set_ylabel('y_scale')
        # ax.set_zlabel('Loss')
        
        # plt.title("Forbenius Loss curve")
        # plt.legend()
        # dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
        # f_name=dt_string+'forb loss curve'
        # plt.savefig(f_name+'.png')

    

