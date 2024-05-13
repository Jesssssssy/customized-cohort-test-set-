# import
import os
import time
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    ScaleIntensity,
    NormalizeIntensity,
    EnsureChannelFirst,
    Resize,
    Compose,
    LoadImage,
    RandFlip,
    RandZoom,
    ToTensor,
    RandRotate,
    Spacing,
    Lambda)
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import lr_scheduler
from torchvision.models import resnet101,ResNet101_Weights

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# training Helper function 1: Define transformation function
def threeD(output):
    return output.repeat(3, 1, 1)


data_transforms = {
    'train': Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        # RandFlip(spatial_axis=1, prob=0.5),  # horizontal
        # RandRotate(prob=0.5, range_x=[-3.14, 3.14]),
        RandZoom(min_zoom=0.99, max_zoom=1.01, prob=0.5),
        # RandAdjustContrast(prob=0.5,gamma=0.6),
        Resize(spatial_size=(224, 224)),
        ToTensor(),
        Lambda(threeD)
    ]),
    'val': Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        Resize(spatial_size=(224, 224)),
        ToTensor(),
        Lambda(threeD)
    ]),
}
y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=2)])


# training Helper function 2: define train_net
# def reset_weights(m):
#     '''
#     Try resetting model weights to avoid
#     weight leakage.
#   '''
#     for layer in m.children():
#         if hasattr(layer, 'reset_parameters'):
#             print(f'Reset trainable parameters of layer = {layer}')
#             layer.reset_parameters()


### Helper function 3:  Define dataset
class MDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

### early stopping
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if __name__ == '__main__':
    # configuration
    batch_size = 50
    num_epochs = 50
    n_split = 10
    loss_function = torch.nn.CrossEntropyLoss()
    auc_metric = ROCAUCMetric()
    best_acc = 0.0

    # figure time spent
    since = time.time()

    # for fold result
    # prepare dataset
    df_path = "/Users/jessy/Documents/PycharmProjects/Combined patch/augmented for combined set.xlsx"
    data_dir = "/Users/jessy/Documents/PycharmProjects/Combined patch/Australian"
    n = 458

    ds_df = pd.read_excel(df_path, sheet_name="Australian new")
    group = ds_df.GROUP
    class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
    num_class = len(class_names)
    image_files = [[os.path.join(data_dir, class_name, x)
                    for x in natsorted(os.listdir(os.path.join(data_dir, class_name))) if x != '.DS_Store']
                   for class_name in class_names]

    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)
    print('Total image count:', num_total)
    print("Label names:", class_names)
    print("Label counts:", [len(image_files[i]) for i in range(num_class)])

    data = [image_file_list[i] for i in range(n)]
    data_label = [image_label_list[i] for i in range(n)]

    # 10 folds
    gkf = StratifiedGroupKFold(n_splits=n_split)
    seeds = [42]
    # random
    for seed in seeds:
        torch.manual_seed(seed)
        # K folds cross validation
        print(f"training for {seed} random seed")
        for fold, (train_indices, test_indices) in enumerate(gkf.split(data, data_label, group)):
            per_fold_train_loss = []
            per_fold_train_acc = []
            per_fold_train_auc = []
            per_fold_val_loss = []
            per_fold_val_acc = []
            per_fold_val_auc = []
            print(f'Fold {fold + 1}')
            print('--------------------------------')
            # Define the data loaders for the train and test sets
            sydney_augmented_diff_indice = []
            HK_augmented_diff_indice = []
            easy = []
            new_train_indices = []
            for i in train_indices:
                if i <= 34:
                    syd_diff_index = [i, 35 + i * 7, 35 + i * 7 + 1, 35 + i * 7 + 2, 35 + i * 7 + 3, 35 + i * 7 + 4,
                                      35 + i * 7 + 5, 35 + i * 7 + 6]
                    sydney_augmented_diff_indice.extend(syd_diff_index)
                if 34 < i <= 38:
                    hk_diff_index = [i + 245, 284 + (i - 35) * 7, 284 + (i - 35) * 7 + 1, 284 + (i - 35) * 7 + 2,
                                     284 + (i - 35) * 7 + 3, 284 + (i - 35) * 7 + 4, 284 + (i - 35) * 7 + 5,
                                     284 + (i - 35) * 7 + 6]
                    HK_augmented_diff_indice.extend(hk_diff_index)
                else:
                    easy.append(i + 273)
            new_train_indices.extend(sydney_augmented_diff_indice)
            new_train_indices.extend(HK_augmented_diff_indice)
            new_train_indices.extend(easy)
            sorted_new_train_indices = natsorted(new_train_indices)

            # organize test indices
            test_indices_new = []
            for j in train_indices:
                if j <= 34:
                    test_indices_new.append(j)
                if 38 >= j > 34:
                    test_indices_new.append(j + 245)
                else:
                    test_indices_new.append(j + 273)
            sorted_test_indices_new = natsorted(test_indices_new)

            # print(sorted_new_train_indices)
            # print("**" * 50)
            # prepare dataset
            data_dir_ = "/Users/jessy/Documents/PycharmProjects/Combined patch/augmented/Australian"
            m = 731

            class_names_ = sorted([x for x in os.listdir(data_dir_) if os.path.isdir(os.path.join(data_dir_, x))])
            num_class_ = len(class_names_)

            image_files_ = [[os.path.join(data_dir_, class_name_, x)
                             for x in natsorted(os.listdir(os.path.join(data_dir_, class_name_))) if x != '.DS_Store']
                            for class_name_ in class_names_]

            image_file_list_ = []
            image_label_list_ = []
            for i, class_name in enumerate(class_names_):
                image_file_list_.extend(image_files_[i])
                image_label_list_.extend([i] * len(image_files_[i]))
            num_total = len(image_label_list_)
            print("Label names:", class_names_)
            print("Label counts:", [len(image_files_[i]) for i in range(num_class_)])

            data_ = [image_file_list_[i] for i in range(m)]
            data_label_ = [image_label_list_[i] for i in range(m)]

            #################################
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sorted_new_train_indices)
            test_sampler = torch.utils.data.sampler.SubsetRandomSampler(sorted_test_indices_new)
            train_loader = DataLoader(MDataset(data_, data_label_, data_transforms['train']), batch_size=batch_size,
                                      sampler=train_sampler)
            val_loader = DataLoader(MDataset(data_, data_label_, data_transforms['val']), batch_size=batch_size,
                                    sampler=test_sampler)
            dataloaders = {'train': train_loader, 'val': val_loader}

            # init cnn each fold
            # Set model parameters -  use the pretrained network, fine tune classifer
            net = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            for param in net.parameters():
                param.requires_grad = False
            net.fc = nn.Linear(in_features=net.fc.in_features, out_features=2,bias=True)
            for param in net.fc.parameters():
                param.requires_grad = True

            model = net

            # init optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.01)
            early_stopper = EarlyStopper(patience=3, min_delta=10)
            # run the training loop for defined number of epochs:
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                # set current loss value
                running_loss = 0.0
                running_corrects = 0
                model.train()  # Set model to training model
                y_pred_train = torch.tensor([], dtype=torch.float32, device=device)
                y_true_train = torch.tensor([], dtype=torch.long, device=device)
                for inputs, labels in dataloaders['train']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    # calculate loss
                    loss = loss_function(outputs, labels)
                    # backward
                    loss.backward()
                    # perform optimization
                    optimizer.step()
                    # print statistics
                    running_loss += loss.item()
                    _, preds = torch.max(outputs.data, 1)  # the max tenser in each row.
                    running_corrects += (preds == labels).sum().item()
                    y_pred_train = torch.cat([y_pred_train, outputs], dim=0)
                    y_true_train = torch.cat([y_true_train, labels], dim=0)

                y_onehot = [y_trans(i) for i in decollate_batch(y_true_train, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred_train)]
                auc_metric(y_pred_act, y_onehot)
                auc = auc_metric.aggregate()
                auc_metric.reset()

                train_loss = running_loss / len(train_loader.sampler)  # return the averaged loss after each epoch.
                # the value mean prob, the second means index.
                train_acc = running_corrects / len(train_loader.sampler) * 100
                print(f'Training loss is {train_loss}, accuracy is {train_acc}, auc is {auc}')

                per_fold_train_loss.append(train_loss)
                per_fold_train_acc.append(train_acc)
                per_fold_train_auc.append(auc)

                # start testing in this epoch
                running_loss_val = 0.0
                running_corrects_val = 0
                total = 0
                with torch.no_grad():
                    y_pred_val = torch.tensor([], dtype=torch.float32, device=device)
                    y_true_val = torch.tensor([], dtype=torch.long, device=device)
                    for inputs, labels in dataloaders['val']:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # generate outputs
                        outputs = model(inputs)
                        # calculate statistics
                        loss = loss_function(outputs, labels)
                        _, preds = torch.max(outputs.data, 1)
                        total += labels.size(0)

                        running_corrects_val += (preds == labels).sum().item()
                        running_loss_val += loss.item()

                        y_pred_val = torch.cat([y_pred_val, outputs], dim=0)
                        y_true_val = torch.cat([y_true_val, labels], dim=0)

                    y_onehot = [y_trans(i) for i in decollate_batch(y_true_val, detach=False)]
                    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred_val)]
                    auc_metric(y_pred_act, y_onehot)
                    auc_val = auc_metric.aggregate()
                    auc_metric.reset()

                    val_loss = running_loss_val / total
                    val_acc = running_corrects_val / total * 100

                    print(f'Validation loss is {val_loss}, accuracy is {val_acc}, auc is {auc_val}')
                    per_fold_val_loss.append(val_loss)
                    per_fold_val_acc.append(val_acc)
                    per_fold_val_auc.append(auc_val)
                    if early_stopper.early_stop(val_loss):
                        break

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))
            ax1.plot(range(num_epochs), per_fold_train_loss, label="train")
            ax1.plot(range(num_epochs), per_fold_val_loss, label="validation")
            ax1.set_title(f"Fold {fold + 1} training and validation loss", fontsize=10)
            ax1.set_xlabel("Epoch", fontsize=9)
            ax1.set_ylabel("Loss", fontsize=9)
            ax1.legend(fontsize=7)

            ax2.plot(range(num_epochs), per_fold_train_acc, label="training")
            ax2.plot(range(num_epochs), per_fold_val_acc, label="validation")
            ax2.set_title(f"Fold {fold + 1} training and validation accuracy", fontsize=10)
            ax2.set_xlabel("Epoch", fontsize=9)
            ax2.set_ylabel("Accuracy", fontsize=9)
            ax2.legend(fontsize=7)

            ax3.plot(range(num_epochs), per_fold_train_auc, label="training")
            ax3.plot(range(num_epochs), per_fold_val_auc, label="valdation")
            ax3.set_title(f"Fold {fold + 1} training and validation auc", fontsize=10)
            ax3.set_xlabel("Epoch", fontsize=9)
            ax3.set_ylabel("AUC", fontsize=9)
            ax3.legend(fontsize=7)

            image_name = f"Performance of fold {fold+1}"
            fig.savefig(image_name)
            # save the model
            save_path = f'./model-fold-{fold+1}.pth'
            torch.save(model.state_dict(), save_path)


            time_elapsed = time.time() - since
            print('Training this fold complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


