import argparse
import os
import numpy as np
import torch
from PIL import Image
from model import DocumentReconstructionModel
from uv_dewarp import dewarp_with_uv
from dataset_loader import get_dataloaders
import torch.nn as nn
from pytorch_msssim import ssim


"""
Evaluation script to compute SSIM of the dewarped output using the model's predicted UV maps.
The model writes to a provided output file the average SSIM across the test set, as well as the individual SSIM values for each image.
The script assumes the dataset is organized in a way compatible with the get_dataloaders function. It loads the dataset with the standard 80:20 train-val split, and then further splits out half of the validation set as a test set. The SSIM metric is computed on this test set.

Arguments:
    - input_dir: Path to the dataset for evaluation (must be compatible with the get_dataloaders function, ie contains an 'rgb' folder and a 'uv' folder). This is optional and will default to 'renders/synthetic_data_pitch_sweep'.
    - weights: Path to the model weights (optional, default: 'best_model.pth').
    - output: File to save the SSIM results (optional, default: './ssim-output.txt').
"""

def main():
    parser = argparse.ArgumentParser(description="Document dewarping inference")
    parser.add_argument('--input_dir', default='renders/synthetic_data_pitch_sweep', help='Path to dataset for evaluation')
    parser.add_argument('--weights',    default='best_model.pth', help='Path to model weights')
    parser.add_argument('--output', default='./ssim-output.txt', help='File to save SSIM results')
    args = parser.parse_args()

    # Configuration
    DATA_DIR = args.input_dir
    BATCH_SIZE = 8
    IMG_SIZE = (512, 512)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    model = DocumentReconstructionModel().to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {args.weights}")

    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_depth=False,  # TODO: Set to True if you want to use depth information
        use_uv=True,     # TODO: Set to True if you want to use UV maps
        use_border=False  # TODO: Set to True if you want to use border masks for better training
    )

    # cut val set in half to create a test set
    val_dataset = val_loader.dataset
    test_size = len(val_dataset) // 2
    test_dataset = torch.utils.data.Subset(val_dataset, range(test_size))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # compute ssim on test set
    ssim_values, ssim_gt, ssim_no_dewarp = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch['rgb'].to(device)  # (B, 3, H, W)
            gt_uv_t = batch['uv']  # (B, 2, H, W)
            filenames = batch['filename'] 
            
            pred_uv_t = model(rgb)  # (B, 2, H, W)

            for i in range(rgb.size(0)):
                filename = filenames[i]

                # need raw rgb for the dewarp
                rgb_raw = np.array(Image.open(os.path.join(DATA_DIR, 'rgb', f'{filename}.jpg')).convert('RGB').resize(IMG_SIZE, Image.BILINEAR))
                rgb_t = torch.from_numpy(rgb_raw).float().permute(2,0,1).unsqueeze(0)
                
                pred_uv = pred_uv_t[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                
                # foreground mask from GT UV
                gt_uv = gt_uv_t[i].numpy().transpose(1, 2, 0)
                fg_mask = ~((gt_uv[:, :, 0] == gt_uv[:, :, 1]) & (gt_uv[:, :, 1] == 0))

                # dewarp
                warped = dewarp_with_uv(rgb_raw, pred_uv, out_size=512, mask=fg_mask)

                # load GT
                gt_img = np.array(Image.open(os.path.join(DATA_DIR, 'ground_truth', f'{filename}.png')).convert('RGB').resize((512, 512), Image.BILINEAR))

                # calc ssim
                warped_t = torch.from_numpy(warped).float().permute(2, 0, 1).unsqueeze(0)
                gt_t = torch.from_numpy(gt_img).float().permute(2, 0, 1).unsqueeze(0)

                warped_gt = dewarp_with_uv(rgb_raw, gt_uv, out_size=512, mask=fg_mask)
                warped_gt_t = torch.from_numpy(warped_gt).float().permute(2, 0, 1).unsqueeze(0)
                ssim_gt.append(ssim(warped_gt_t, gt_t, data_range=255).item()) # SSIM of GT dewarp vs GT for reference
                ssim_no_dewarp.append(ssim(rgb_t, gt_t, data_range=255).item())
                ssim_values.append(ssim(warped_t, gt_t, data_range=255).item()) # TODO: what is data range here?
    
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
    print(f"No dewarp SSIM: {np.mean(ssim_no_dewarp):.4f}")
    print(f"GT UV SSIM: {np.mean(ssim_gt):.4f}")
    print(f"N images: {len(ssim_values)}")

    
    with open(args.output, 'w') as f:
        f.write(f"Average SSIM: {np.mean(ssim_values)}\n")
        f.write("SSIM values for each image:\n")
        for s in ssim_values:
            f.write(f"{s}\n")



if __name__ == "__main__":
    main()