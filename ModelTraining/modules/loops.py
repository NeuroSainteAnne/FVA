import torch
import wandb
from .figure import create_figure
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

eps = 10e-5
        
# Function for the training loop
def training_loop(status):
    #### TRAINING LOOP
    running_loss = 0.0
    running_slices = 0
    status.model.train()  # Set model to training mode
    print("Training epoch", status.epoch)
    
    # Iterate over the training dataset
    for i, data in enumerate(status.train_dataloader, 0):
        # Initialize metrics and reset parameter gradients
        train_metrics = {"loss": None, "loss_mask": None, "loss_coarse": None, "loss_viz": None, "loss_noviz": None}
        status.optimizer.zero_grad()
        
        # Load input data and labels to the device
        x = data[0].to(status.device)
        y = data[3].to(status.device)
        mask = data[1].to(status.device).type(torch.bool)
        n = x.shape[0]
        
        # Forward pass to get model outputs
        outputs = status.model(x)
        
        # Calculate mask loss
        loss_mask = torch.mean(status.bce(outputs[:, 0], mask[:, 0].float()))
        loss = status.config["lambda_mask"] * loss_mask
        train_metrics["loss_mask"] = loss_mask.item()
        
        # Calculate coarse segmentation loss if the mask is present
        skullmask = mask[:, 0] > 0.5
        if torch.sum(skullmask) > 0:
            loss_coarse = torch.mean(status.bce(
                torch.masked_select(outputs[:, status.blob_index + 1], skullmask),
                torch.masked_select(y[:, status.blob_index].float(), skullmask)
            ))
            
            if status.config["lambda_coarse_segm"] > 0:
                loss += status.config["lambda_coarse_segm"] * loss_coarse
            train_metrics["loss_coarse"] = loss_coarse.item()
            
            # Calculate visualization loss if configured
            if torch.sum(skullmask) > 0:
                if status.config["lambda_viz"] > 0:
                    loss_viz = torch.mean(
                        status.bce_weighted(
                            torch.masked_select(outputs[:, status.viz_index + 1], skullmask),
                            torch.masked_select(y[:, status.viz_index].float(), skullmask)
                        )
                    )
                    loss += status.config["lambda_viz"] * loss_viz
                    train_metrics["loss_viz"] = loss_viz.item()
        
        # Backpropagate the loss and update model parameters
        if loss.item() < 0:
            assert False  # Assert no negative loss
        else:
            loss.backward()
            status.optimizer.step()
            train_metrics["loss"] = loss.item()
            train_metrics = {i: j for i, j in train_metrics.items() if j is not None}
            
            # Log training metrics to Weights & Biases (if applicable)
            if status.run is not None:
                wandb.log({"train": train_metrics})
            running_loss += loss.item()
            running_slices += n
        
        # Conditions for debugging
        if status.run is None and running_slices > 128:
            break
        if i * status.config["train_batch_size"] > status.config["slices_per_epoch"]:
            break

# Function to stabilize batch normalization running mean and variance
def bn_loop(status):
    with torch.no_grad(): # Disable gradient calculation to save memory and computation
        # Iterate over the batch normalization dataset
        for i, data in enumerate(status.bn_dataloader, 0):
            x = data[0].to(status.device)
            # Perform forward pass to update batch normalization statistics
            status.model.forward(x)
            if i >= 20:
                break

def validation_loop(status):
    this_auc = -1.0
    this_best_kappa = -1.0
    
    # Dictionaries to store validation measures and metrics
    val_measures = {"skull_VP":[],"skull_VN":[],"skull_FP":[],"skull_FN":[],
                    "fine_VP":[],"fine_VN":[],"fine_FP":[],"fine_FN":[],
                    "blob_VP":[],"blob_VN":[],"blob_FP":[],"blob_FN":[],
                    "viz_VP":[],"viz_VN":[],"viz_FP":[],"viz_FN":[],
                    "noviz_VP":[],"noviz_VN":[],"noviz_FP":[],"noviz_FN":[],
                    "all_viz":[]}
    val_metrics = {"mask_dice":[],"coarse_dice":[],"viz_dice":[],"noviz_dice":[], "percent_correct":[],
                   "coarse_accuracy":[],"viz_accuracy":[],"noviz_accuracy":[], "loss":[],
                  "loss_mask":[],"loss_coarse":[],"loss_viz":[],"loss_noviz":[]}
    pred_classif = []
    true_classif = []
    pat_index = []
    
    print("Validation at epoch", status.epoch)
    status.model.eval() # Validation mode
    with torch.no_grad():
        # Generate and log figures for visualization
        for viz_type, name in zip([status.simple_viz, status.simple_noviz, status.simple_mixviz], ["viz", "noviz", "mixviz"]):
            fig = create_figure(viz_type, status)
            if fig is not None:
                if status.run is None:
                    fig.show()
                else:
                    wandb.log({f"val.img.{name}": wandb.Image(plt)}, commit=False)
                fig.clear()
                plt.clf()
                plt.close()
            
        # Iterate over the validation dataset
        for i, data in enumerate(status.valid_dataloader, 0):
            # Load input data, labels, and mask to the device
            x = data[0].to(status.device)
            y = data[3].to(status.device)
            mask = data[1].to(status.device).type(torch.bool)
            
            # Perform inference to get model predictions
            outputs = status.model(x)
            n = x.shape[0]
            pat_index += [data[2][:,0].numpy()]
            
            # Compute metrics for skull
            skull_GT = mask[:,0]>0.5
            skull_Pred = outputs[:,0]>0
            val_measures["skull_VP"] += list((skull_GT*skull_Pred).view(n,-1).sum(1).cpu().numpy())
            val_measures["skull_VN"] += list((~skull_GT*~skull_Pred).view(n,-1).sum(1).cpu().numpy())
            val_measures["skull_FP"] += list((~skull_GT*skull_Pred).view(n,-1).sum(1).cpu().numpy())
            val_measures["skull_FN"] += list((skull_GT*~skull_Pred).view(n,-1).sum(1).cpu().numpy())
            
            # Additional metrics for blob and visualization predictions
            blob_GT = skull_GT*(y[:,status.blob_index]>0.5)
            blob_Pred = skull_Pred*(outputs[:,status.blob_index+1]>0)
            val_measures["blob_VP"] += list((blob_GT*blob_Pred).view(n,-1).sum(1).cpu().numpy())
            val_measures["blob_VN"] += list((~blob_GT*~blob_Pred*skull_GT).view(n,-1).sum(1).cpu().numpy())
            val_measures["blob_FP"] += list((~blob_GT*blob_Pred*skull_GT).view(n,-1).sum(1).cpu().numpy())
            val_measures["blob_FN"] += list((blob_GT*~blob_Pred).view(n,-1).sum(1).cpu().numpy())
            
            # Calculate metrics for flairviz
            viz_GT = blob_GT*(y[:,status.viz_index]>0.5)
            viz_Pred = blob_Pred*(outputs[:,status.viz_index+1]>0)
            val_measures["viz_VP"] += list((viz_GT*viz_Pred).view(n,-1).sum(1).cpu().numpy())
            val_measures["viz_VN"] += list((~viz_GT*~viz_Pred*skull_GT).view(n,-1).sum(1).cpu().numpy())
            val_measures["viz_FP"] += list((~viz_GT*viz_Pred*skull_GT).view(n,-1).sum(1).cpu().numpy())
            val_measures["viz_FN"] += list((viz_GT*~viz_Pred).view(n,-1).sum(1).cpu().numpy())
            noviz_GT = blob_GT*(y[:,status.viz_index]<0.5)
            noviz_Pred = blob_Pred*(outputs[:,status.viz_index+1]<0)
            val_measures["noviz_VP"] += list((noviz_GT*noviz_Pred).view(n,-1).sum(1).cpu().numpy())
            val_measures["noviz_VN"] += list((~noviz_GT*~noviz_Pred*skull_GT).view(n,-1).sum(1).cpu().numpy())
            val_measures["noviz_FP"] += list((~noviz_GT*noviz_Pred*skull_GT).view(n,-1).sum(1).cpu().numpy())
            val_measures["noviz_FN"] += list((noviz_GT*~noviz_Pred).view(n,-1).sum(1).cpu().numpy())
            composite_l = torch.sigmoid(outputs[:,0]) + \
                           (torch.sigmoid(outputs[:,status.blob_index+1])*torch.sigmoid(outputs[:,0])) + \
                           (torch.sigmoid(outputs[:,status.viz_index+1])*torch.sigmoid(outputs[:,status.blob_index+1])*torch.sigmoid(outputs[:,0]))
            viz_Pred_compo = composite_l > 2.5
            val_measures["all_viz"] += list(viz_Pred_compo.view(n,-1).sum(1).cpu().numpy())
        
            # Losses computation
            loss_mask = status.bce(outputs[:,0],mask[:,0].float()).mean()*n/status.config["test_batch_size"]
            loss = status.config["lambda_mask"]*loss_mask
            val_metrics["loss_mask"] += [loss_mask.item()]
            loss_coarse = status.bce(outputs[:,status.blob_index+1],blob_GT.float()).masked_select(skull_GT).mean()*n/status.config["test_batch_size"]
            val_metrics["loss_coarse"] += [loss_coarse.item()]
            if status.config["lambda_coarse_segm"] > 0:
                loss += status.config["lambda_coarse_segm"]*loss_coarse
            loss_viz = status.bce_weighted(outputs[:,status.viz_index+1],viz_GT.float()).masked_select(skull_GT).mean()*n/status.config["test_batch_size"]
            val_metrics["loss_viz"] += [loss_viz.item()]
            if status.config["lambda_viz"] > 0:
                loss += status.config["lambda_viz"]*loss_viz
            val_metrics["loss"] += [loss.item()]
            
            # Compute accuracy for different classes 
            status.acc.update(torch.masked_select(outputs[:,status.blob_index+1],skull_GT),
                        torch.masked_select(blob_GT.float(),skull_GT))
            val_metrics["coarse_accuracy"] += [status.acc.compute().item()*n/status.config["test_batch_size"]]
            status.acc.update(torch.masked_select(outputs[:,status.viz_index+1]*blob_Pred,skull_GT),
                        torch.masked_select(viz_GT.float(),skull_GT))
            val_metrics["viz_accuracy"] += [status.acc.compute().item()*n/status.config["test_batch_size"]]
            status.acc.update(torch.masked_select((-1.0*outputs[:,status.viz_index+1])*blob_Pred,skull_GT),
                        torch.masked_select(noviz_GT.float(),skull_GT))
            val_metrics["noviz_accuracy"] += [status.acc.compute().item()*n/status.config["test_batch_size"]]
            sum_blob_GT = blob_GT.sum().item()
            
            # slice classification accuracy 
            if sum_blob_GT > 0:
                val_metrics["percent_correct"] += [torch.sum(torch.eq(
                                     torch.masked_select(viz_Pred,blob_GT),
                                     torch.masked_select(viz_GT,blob_GT)
                                 )).item()/sum_blob_GT*n/status.config["test_batch_size"]]
                
            # Automatic slice classification
            for j in range(n):
                noise_thr = 20
                class_thr = 0.2
                if torch.sum(skull_Pred[j]) < noise_thr:
                    pred_classif += [0]
                elif torch.sum(blob_Pred[j]) < noise_thr:
                    pred_classif += [0]
                else:
                    ratio = torch.sum(viz_Pred[j])/(torch.sum(blob_Pred[j]))
                    if ratio > 1-class_thr:
                        pred_classif += [3]
                    elif ratio < class_thr:
                        pred_classif += [1]
                    else:
                        pred_classif += [2]
                if torch.sum(skull_GT[j]) < noise_thr:
                    true_classif += [0]
                elif torch.sum(blob_GT[j]) < noise_thr:
                    true_classif += [0]
                else:
                    ratio = torch.sum(viz_GT[j])/torch.sum(blob_GT[j])
                    if ratio > 1-class_thr:
                        true_classif += [3]
                    elif ratio < class_thr:
                        true_classif += [1]
                    else:
                        true_classif += [2]
            if status.run is None and i > 2:
                break
                
    pat_index = np.concatenate(pat_index)
    val_measures = {key:np.array(value) for key, value in val_measures.items()}
    val_metrics = {i:np.mean(j) for i, j in val_metrics.items()}
    
    # Dice measurement
    val_metrics["mask_dice"] = (2*val_measures["skull_VP"].sum()+eps) / \
                                 (2*val_measures["skull_VP"].sum()+val_measures["skull_FP"].sum()+val_measures["skull_FN"].sum()+eps)
    val_metrics["coarse_dice"] = (2*val_measures["blob_VP"].sum()+eps) / \
                                 (2*val_measures["blob_VP"].sum()+val_measures["blob_FP"].sum()+val_measures["blob_FN"].sum()+eps)
    val_metrics["viz_dice"] = (2*val_measures["viz_VP"].sum()+eps) / \
                                 (2*val_measures["viz_VP"].sum()+val_measures["viz_FP"].sum()+val_measures["viz_FN"].sum()+eps)
    val_metrics["noviz_dice"] = (2*val_measures["noviz_VP"].sum()+eps) / \
                                 (2*val_measures["noviz_VP"].sum()+val_measures["noviz_FP"].sum()+val_measures["noviz_FN"].sum()+eps)
    val_metrics["viz_global_dice"] = (val_metrics["viz_dice"]+val_metrics["noviz_dice"])/2
    val_metrics["kappa_slice_classif"] = cohen_kappa_score(true_classif, pred_classif)

    # Patient-wise binary classification
    true_classif_binary = np.array(true_classif)>=2
    gt_patient = []
    pred_voxels = []
    for p in list(set(list(pat_index))):
        gt_patient.append(np.sum(true_classif_binary[pat_index==p]) > 0)
        pred_voxels.append(np.sum(val_measures["all_viz"][pat_index==p]))

    # AUC computation (patient)
    try:
        this_auc = roc_auc_score(gt_patient,pred_voxels)
    except ValueError:
        this_auc=0
    if status.run is not None: wandb.log({"val.patwise.auc": this_auc}, commit=False)
    kt = np.arange(0,4000,1)
    kv = []
    for i in kt:
        kv.append(cohen_kappa_score(gt_patient,np.array(pred_voxels)>i))
        
    # Kappa computation (patient)
    this_best_kappa = np.max(kv)
    if status.run is not None: wandb.log({"val.patwise.bestkappa": this_best_kappa}, commit=False)
    plt.plot(kt,kv)
    if status.run is not None: wandb.log({"val.patwise.kappacurve.img": wandb.Image(plt)}, commit=False)
    plt.clf()
    plt.close()
    
    # Kappa figure (patient)
    fig = sns.heatmap(pd.crosstab(gt_patient,
                                 np.array(pred_voxels)>kt[np.argmax(kv)], 
                                 rownames=['Pred'], colnames=['GT_Binary'], dropna=False),
                    square=True, annot=True, fmt='g')
    if status.run is not None: wandb.log({"val.patwise.bestkappa.img": wandb.Image(plt)}, commit=False)
    plt.clf()
    plt.close()
    
    # Kappa figure (slice)
    fig = sns.heatmap(pd.crosstab( pd.Categorical(true_classif),pd.Categorical(pred_classif), 
                                  rownames=["True"], colnames=["Pred"]), 
                      square=True, annot=True, fmt='g')
    if status.run is not None: wandb.log({"val.img.classif_slice": wandb.Image(plt)}, commit=False)
    if status.run is not None: fig.clear()
    plt.clf()
    plt.close()

    # Update scheduler
    if status.config["lr_scheduler"]:
        if status.config["lr_scheduler_mode"] == "plateau":
            val_metrics["learning_rate"] = status.optimizer.param_groups[0]['lr']
            status.scheduler.step(val_metrics["loss"])
        else:
            val_metrics["learning_rate"] = status.scheduler.get_last_lr()[0]
            status.scheduler.step()
        
    # Log metrics
    if status.run is not None: wandb.log({"val":val_metrics, "epoch": status.epoch}, commit=True)

    return val_metrics["loss"], val_metrics["viz_global_dice"], this_best_kappa, this_auc
