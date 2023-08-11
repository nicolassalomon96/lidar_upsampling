import torch.nn
import torch.optim as optim
import kaolin
import time
from unet_model import *
from data_gen_distance import *
from pointcloud_utils_functions_v2 import *

##################################### VARIABLES #####################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
lr = 0.01
epoch_number = 0
EPOCHS = 100
best_vloss = 1_000_000

##################################### DATASET - DATALOADER ##########################################

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()

train_dataset = IterDataset(train_data_generator)
valid_dataset = IterDataset(valid_data_generator)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

##################################### NETWORK #######################################################

Unet = UNet()
Unet = Unet.to(device)

###################################### LOSS #########################################################

class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        self.device = device

    def forward(self, image_pred, image_gt):
        """
        Compute the Chamfer distance between predicted and ground truth point clouds.

        Args:
            point_cloud_pred (torch.Tensor): Predicted point cloud tensor of shape (batch_size, num_points, num_dims).
            point_cloud_gt (torch.Tensor): Ground truth point cloud tensor of shape (batch_size, num_points, num_dims).

        Returns:
            torch.Tensor: Chamfer distance loss.
        """
        pointcloud_pred = range_image_to_pointcloud_pytorch(image_pred * kitti_max_distance, device)
        pointcloud_gt = range_image_to_pointcloud_pytorch(image_gt * kitti_max_distance, device)

        chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(pointcloud_pred[:,:,:3].to(self.device), pointcloud_gt[:,:,:3].to(self.device))
        chamfer_loss_mean = torch.mean(chamfer_loss)

        return chamfer_loss_mean

loss_fn = ChamferLoss(device=device)
#loss_fn = nn.L1Loss()

############################## OPTIMIZER - LR SCHEDULER ##############################################

optimizer = optim.Adam(Unet.parameters(), lr=lr, weight_decay=lr/100)
#optimizer = torch.optim.Rprop(Unet.parameters(), lr=lr, weight_decay=lr/100)
#optimizer.param_groups[0]['initial_lr'] = lr
#optimizer = torch.optim.SGD(mlp_net.parameters(), lr=lr)
#optimizer = torch.optim.RMSprop(mlp_net.parameters(), lr=lr)
#optimizer = torch.optim.LBFGS(mlp_net.parameters(), lr=lr, max_iter=10)#, history_size=100)

#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5, last_epoch=20) #Cada 10 épocas lr = lr * gamma
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=0.01)
#lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.01, max_lr=lr, step_size_up=10, cycle_momentum=False)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=0.01)
#iters = len(train_urls) 

############################## TRAIN LOOP #############################################

def train_one_epoch(epoch_index):
    running_loss = 0
 
    for j, data in enumerate(train_dataloader):
        #print(f'Batch: {j} / {np.floor(len(train_urls)/batch_size)}')
        # Every data instance is an input + label pair
        lrimgs, hrimgs = data
        lrimgs = lrimgs.to(device)
        hrimgs = hrimgs.to(device)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        #Get network outputs
        outputs = Unet(lrimgs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, hrimgs)
        #print(loss)

        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
    return running_loss / (j + 1)

#def train():

for epoch in range(EPOCHS):
    inicio = time.time()
    Unet.train()
    avg_loss = train_one_epoch(epoch)

    # We don't need gradients on to do reporting
    running_vloss = 0.0
    #Unet.eval()

    with torch.no_grad():
        for k, vdata in enumerate(valid_dataloader):
            vlrimgs, vhrimgs = vdata
            vlrimgs, vhrimgs = vlrimgs.to(device), vhrimgs.to(device)
            
            voutputs = Unet(vlrimgs)               

            vloss = loss_fn(voutputs, vhrimgs)
            running_vloss += vloss.item()
            
        avg_vloss = running_vloss / (k + 1)
        lr_scheduler.step(avg_vloss)
        #lr_scheduler.step()
    fin = time.time()
    print(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos - Estimado: {((EPOCHS-epoch_number-1)*(fin-inicio))/60.0} minutos')
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\unet\6_pytorch_implementation\model\Unet_Chamfer_Adam_ep{epoch_number+1}.pth'
        torch.save(Unet.state_dict(), model_path)

    epoch_number += 1

model_path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\unet\6_pytorch_implementation\model\Unet_Chamfer_Adam_ep{epoch_number}.pth'
torch.save(Unet.state_dict(), model_path)

#if __name__ == "__main__":
#    train()