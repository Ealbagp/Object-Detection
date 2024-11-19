import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
sys.path.append('..')
import module.dataloader as dataloader
from tqdm import tqdm
from model_architecture import Network
from model_architecture_improved import NetworkImproved, CustomCNN


PROPOSAL_SIZE = (128, 128)
BATCH_SIZE = 100
BALANCE = 0.5

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),    # Convert PIL Image to Tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor
                       std=[0.229, 0.224, 0.225])
])

normalize_only = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.ToTensor(),    # Convert PIL Image to Tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor
                       std=[0.229, 0.224, 0.225])
])


dataset_train = dataloader.PotholeDataset(
    '../Potholes/annotated-images/',
    '../Potholes/labeled_proposals/',
    '../Potholes/annotated-images/',
    transform=transform,
    proposals_per_batch=BATCH_SIZE,
    proposal_size=PROPOSAL_SIZE,
    balance=BALANCE,
    split='train'
)

dataset_val = dataloader.PotholeDataset(
    '../Potholes/annotated-images/',
    '../Potholes/labeled_proposals/',
    '../Potholes/annotated-images/',
    transform=normalize_only, 
    proposals_per_batch=BATCH_SIZE,
    proposal_size=PROPOSAL_SIZE,
    balance=BALANCE,
    split='val'
)
# dataset_test = dataloader.PotholeDataset('../Potholes/annotated-images/', '../Potholes/labeled_proposals/', '../Potholes/annotated-images/', proposals_per_batch=BATCH_SIZE, proposal_size=PROPOSAL_SIZE, balance=BALANCE, split='test')


train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)



model = Network(proposal_size=PROPOSAL_SIZE)
model.apply(initialize_weights)
model.to(device)
#Initialize the optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
# Initialize the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05)





def train(model, optimizer, lr_scheduler, num_epochs=10):
    
    
    def loss_fun(output, target):
        pos_weight = torch.tensor([4.0]).to(device)
        return F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
    
    out_dict = {
              'train_acc': [],
              'val_acc': [],
              'train_loss': [],
              'val_loss': []}
  
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []
        # for minibatch_no, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
        for idx, (single_image_dict) in enumerate(train_loader):
            # for proposal, label, proposal_image in zip(single_image_dict['proposals'], single_image_dict['labels'], single_image_dict['proposal_images']):
            proposal_image, label = single_image_dict['proposal_images'][0].to(device), single_image_dict['labels'][0].to(device)
            label = label.unsqueeze(1).float()
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(proposal_image)
            #Compute the loss
            loss = loss_fun(output, label)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            output = nn.functional.sigmoid(output)
            predicted = output > 0.5
            train_correct += (label==predicted).sum().cpu().item() / len(label)

        lr_scheduler.step()
        
        #Comput the test accuracy
        val_loss = []
        val_correct = 0
        model.eval()
        for single_val_dict in val_loader:
            # for proposal_val, label_val, proposal_image_val in zip(single_val_dict['proposals'], single_val_dict['labels'], single_val_dict['proposal_images']):
            proposal_image_val, label_val = single_val_dict['proposal_images'][0].to(device), single_val_dict['labels'][0].to(device)
            label_val = label_val.unsqueeze(1).float()
            
            with torch.no_grad():
                output = model(proposal_image_val)

            val_loss.append(loss_fun(output, label_val).cpu().item())
            output = nn.functional.sigmoid(output)
            predicted = output > 0.5
            val_correct += (label_val==predicted).sum().cpu().item() / len(label_val)

        out_dict['train_acc'].append(train_correct/len(dataset_train))
        out_dict['val_acc'].append(val_correct/len(dataset_val))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))

        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(val_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['val_acc'][-1]*100:.1f}%") # Dividing by 5 because of the batch_size
        
    return out_dict


train(model, optimizer, lr_scheduler, num_epochs=30)

# Save the model
torch.save(model.state_dict(), 'model 3.pth')