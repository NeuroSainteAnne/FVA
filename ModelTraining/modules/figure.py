import torch
import matplotlib.pyplot as plt

### FIGURE CREATION function
# This function generates figures for model predictions and ground truth for a given dataset.
def create_figure(fig_dataset, status):
    # Set the model to evaluation mode to disable dropout and other training-specific layers
    status.model.eval()
    with torch.no_grad():
        try:
            # Get the next batch of data from the dataset for visualization
            simple_data = next(enumerate(fig_dataset))[1]
        except StopIteration:
            # If no data is available, return None
            return None
        
        # Load the input data to the appropriate device (e.g., CPU or GPU)
        x = simple_data[0].to(status.device)
        
        # Perform inference with the model to get the predictions
        if status.config["input_2.5D"] == "smp3d":
            # For 2.5D input, forward pass and select the center z-slice
            simple_outputs = status.model.forward(x)
            x = x[:, :, status.zcenter]
        else:
            # For regular input, forward pass
            simple_outputs = status.model.forward(x)
        
        # Number of examples in the batch
        simple_size = simple_outputs.shape[0]
        n_rows = 3  # Number of rows in the figure (DWI, Computed, Ground truth)
        
        # Create a figure for the current batch of data
        fig = plt.figure(figsize=(simple_outputs.shape[0] * 2, n_rows * 2)) 
        
        # Loop through each example in the batch and generate the corresponding subplots
        for i in range(simple_size):
            # Extract true image, predicted mask, and ground truth mask
            true_diff = x[i, 0].cpu().numpy().T[::-1]  # Transpose and flip for visualization
            pred_mask = torch.sigmoid(simple_outputs[i, status.mask_index]).cpu().numpy().T[::-1]  # Apply sigmoid activation
            true_mask = simple_data[1][i, 0].numpy().T[::-1]
            
            # Extract coarse stroke and flairviz predictions and ground truth
            pred_coarse = torch.sigmoid(simple_outputs[i, status.blob_index]).cpu().numpy().T[::-1] * (pred_mask > 0.5)
            true_coarse = simple_data[3][i, status.blob_index_ydat].numpy().T[::-1]
            pred_viz = torch.sigmoid(simple_outputs[i, status.viz_index]).cpu().numpy().T[::-1] * (pred_coarse > 0.5)
            true_viz = simple_data[3][i, status.viz_index_ydat].numpy().T[::-1]
            
            # Create composite images for visualization
            true_composite = true_mask + (true_coarse * true_mask) + (true_viz * true_coarse * true_mask)
            pred_composite = (pred_mask > 0.5).astype(float) + (pred_coarse > 0.5).astype(float) + (pred_viz > 0.5).astype(float)
            
            # Create subplot for the true diffusion-weighted image (DWI)
            ax1 = fig.add_subplot(n_rows, simple_size, i + 1)
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
            ax1.imshow(true_diff, cmap="gray", vmin=-3, vmax=3)
            plt.title(status.pat_ids[status.pat_ids.index == int(simple_data[2][i, 0])]["Patient"].iloc[0], fontsize=8)
            if i == 0: plt.ylabel("DWI")
            
            # Create subplot for the computed (predicted) composite image
            ax2 = fig.add_subplot(n_rows, simple_size, i + 1 + (simple_size * 1))
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
            ax2.imshow(pred_composite, vmin=0, vmax=3)
            if i == 0: plt.ylabel("Computed")
            
            # Create subplot for the ground truth composite image
            ax3 = fig.add_subplot(n_rows, simple_size, i + 1 + (simple_size * 2))
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
            if i == 0: plt.ylabel("Ground truth")
            ax3.imshow(true_composite, vmin=0, vmax=3)
    
    # Return the generated figure
    return fig