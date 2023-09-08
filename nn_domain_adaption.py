'''
- pca on source : X_pca_src
- pca on target : X_pca_target
- cov of X_pca_src : Cov_src_pca
- cov of X_pca_target : Cov_target_pca
- input v_n : X_pca_src
- output y_n : X_pca_src@Cov_target_pca
- Use layer 1 weights : X_pca_src@layer1_weights
'''



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,precomputed_weights):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Freeze the weights of layer 2
        self.fc2.weight.requires_grad = False
        # Assign precomputed weights to the weights of layer 2
        self.fc2.weight.data = torch.from_numpy(precomputed_weights).float()

        # Tie the weights of the first and last hidden layers
        self.fc3.weight = nn.Parameter(self.fc1.weight.t())


    def forward(self, x):
        x = self.fc1(x)
        # print("1st layer",x)
        x = self.fc2(x)
        # print("2nd layer",x)
        x = self.fc3(x)
        # print("3rd layer",x)
        return x

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
    plt.show()
    
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

    pis = np.array([0.5, 0.5])
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


if __name__=='__main__':
    idx=datetime.now()
    generate_data=False
    if (generate_data):
        # Generate data
        n=1000
        # source domain: GMM - class 0
        src_mus_c0 = [np.array([-2, 2]), np.array([-2,-2])]
        src_covs_c0 = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
        src_c0 = gaussian_mixture_model(src_mus_c0,src_covs_c0,n)

        # source domain: GMM - class 1
        src_mus_c1 = [np.array([2, 2]), np.array([2, -2])]
        src_covs_c1 = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.8], [-0.8, 1]])]
        src_c1 = gaussian_mixture_model(src_mus_c1,src_covs_c1,n)
        
        homography_mat=np.round(np.random.uniform(-1,1,(2,2)),decimals=5) #2x2 #src x T = target

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
        src_data=np.load('Src.npy')
        target_data=np.load('Target.npy')
    
    plot(src_data,target_data,'Sampled Data',idx)
    
    # Save dataset (optional)
    save_flag=False
    if (save_flag):
        np.save("Src.npy",src_data)
        np.save("Target.npy",target_data)

    # pca on source : X_pca_src
    X_pca_src,X_s_ori,_ = pca(src_data) #X_s: Dxd (column vector)

    # pca on target : X_pca_target
    X_pca_target,X_t_ori,_ = pca(target_data) #X_t: Dxd (column vector)

    plot(X_pca_src,X_pca_target,'PCA_space',idx)

    # cov of X_pca_src : Cov_src_pca
    Cov_src_pca=np.cov(X_pca_src.T)
    # print("Cov_src_pca",Cov_src_pca)

    # cov of X_pca_target : Cov_target_pca
    Cov_target_pca=np.cov(X_pca_target.T)

    # Generate v_n spanning src domain in PCA space
    v_n_points=2000
    kde_flag=True
    if kde_flag:
        # density estimation
        # use grid search cross-validation to optimize the bandwidth
        params = {"bandwidth": np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(X_pca_src)

        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_

        # sample new points from the data
        X_pca_src_newData = kde.sample(v_n_points, random_state=0)
    else:
        #uniform generation
        X_pca_src_newData=[]
        for i in range(X_pca_src.shape[1]):
            new_sample_1d = np.random.uniform(np.min(X_pca_src[:,i]),np.max(X_pca_src[:,i]),v_n_points)
            X_pca_src_newData.append(new_sample_1d)
        X_pca_src_newData=np.column_stack(X_pca_src_newData)

    plot(X_pca_src_newData,X_pca_src_newData,"new sampling",idx)
    # input v_n : X_pca_src
    input_data=torch.from_numpy(X_pca_src_newData).float()

    # output y_n : X_pca_src@Cov_target_pca
    output_data=torch.from_numpy(X_pca_src_newData@Cov_target_pca).float()

    # Initialize the model
    d=2
    input_size = d
    hidden_size = d  # You can adjust this value based on your requirements
    output_size = d

    # Define the decay rates to try
    decay_rates = [1e-5,1e-4, 1e-3, 1e-2,1e-1]

    best_model = None
    best_loss = float('inf')
    best_decay_rate = None
    best_F_norm = None

    #containers of loss & Forbenius norm
    loss_array = []
    F_norm_array = []
    exp_dict=dict() #'epoch':{'mse_array':[],'F_norm_array':[]}

    # Loop through each decay rate and train the model
    for decay_rate in decay_rates:
        print("Initializing model with {} decay rate".format(decay_rate))
        model = NeuralNetwork(input_size, hidden_size, output_size,Cov_src_pca)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=decay_rate)

        # Training loop
        num_epochs = 100  # You can adjust this value based on your requirements
        for epoch in range(num_epochs):
            # print("Epoch : ",epoch)
            # Forward pass
            output = model(input_data)

            # Compute the loss
            loss = criterion(output_data, output)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss every 100 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # Evaluate the trained model
        model.eval()
        with torch.no_grad():
            output = model(input_data)
            loss = criterion(output_data, output)
            print(f"Final Loss: {loss.item()}")

        # Use layer 1 weights : X_pca_src@layer1_weights
        src_pca_aligned=X_pca_src@model.fc1.weight.cpu().detach().numpy()

        # check Forbenius norm
        # cov of src_aligned : Cov_src_pca_aligned
        Cov_src_pca_aligned = np.cov(src_pca_aligned.T)
        forb_norm = np.sum(np.square(Cov_src_pca_aligned-Cov_target_pca))
        print("Forb_norm: ", forb_norm)

         # Compare the loss to find the best model
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_decay_rate = decay_rate
            best_F_norm = forb_norm
        
        loss_array.append(loss)
        F_norm_array.append(forb_norm)
    
    # Print the best decay rate , loss and Forbenius norm
    print(f"Best Decay Rate: {best_decay_rate}, Best Loss: {best_loss}, Best Forbenius norm: {best_F_norm}")

    # Use the best model to obtain the aligned source data
    best_model.eval()
    with torch.no_grad():
        src_pca_aligned = X_pca_src @ best_model.fc1.weight.cpu().detach().numpy()

    plt.scatter([-5,-4,-3,-2,-1],loss_array,color='r',label='MSE loss')
    plt.scatter([-5,-4,-3,-2,-1],F_norm_array,color='b',label='Forbenius norm')
    plt.title("Loss curve")
    plt.ylabel('Loss')
    plt.xlabel('Decay rates')
    plt.legend()
    dt_string = idx.strftime('%d_%m_%Y_%H_%M_%S')
    f_name=dt_string+'loss curve'
    plt.savefig(f_name+'.png')
    plt.show()

    plot(src_pca_aligned,X_pca_target,'aligned',idx)

    plot(src_pca_aligned,X_pca_src,'aligned Vs Before',idx)

    plot(src_pca_aligned,src_pca_aligned,'Only aligned',idx)
