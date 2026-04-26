import argparse
import os
import torch
# from model import DocumentDewarpNet, SimpleEncoderDecoder
from dataset_loader import DocumentReconstructionModel
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_msssim import ssim as ssim_func
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Document Dewarping Inference')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output folder')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--ground_truth_dir', type=str, default=None) # TODO: is this allowed or do we need to figure out where this comes from
    args = parser.parse_args()

    # TODO: naming conventions are confusing - what is expected? is ground truth ssim calc supposed to be here or part of the model processing?

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    # model = DocumentDewarpNet().to(device)
    model = DocumentReconstructionModel(model_type='m3').to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print("==> Model loaded successfully")

    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Find all input files
    input_files = sorted(
        glob.glob(os.path.join(args.input_dir, '*.png')) +
        glob.glob(os.path.join(args.input_dir, '*.jpg'))
    )
    print(f"Found {len(input_files)} images")

    ssim_scores = []

    for input_path in tqdm(input_files, desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # Extract just the N part from crumpled_N
        n = base_name.replace('crumpled_', '')

        # Load and preprocess
        img = Image.open(input_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Inference
        with torch.no_grad():
            rectified, flow = model(img_tensor)

        # Denormalize rectified output
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        rectified = (rectified.squeeze(0) * std + mean).clamp(0, 1)

        # Save rectified image
        rectified_np = (rectified.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rectified_np).save(os.path.join(args.output_dir, f'rectified_{n}.png'))

        # Save UV visualization
        flow_np = flow.squeeze(0).cpu().numpy()  # [2, H, W]
        flow_np = (flow_np - flow_np.min()) / (flow_np.max() - flow_np.min() + 1e-8)
        uv_vis = np.zeros((flow_np.shape[1], flow_np.shape[2], 3), dtype=np.uint8)
        uv_vis[:, :, 0] = (flow_np[0] * 255).astype(np.uint8)  # R = x flow
        uv_vis[:, :, 1] = (flow_np[1] * 255).astype(np.uint8)  # G = y flow
        Image.fromarray(uv_vis).save(os.path.join(args.output_dir, f'predicted_uv_{n}.png'))

         # Compute SSIM if ground truth provided
        if args.ground_truth_dir:
            gt_path = os.path.join(args.ground_truth_dir, f'{base_name}.png')
            if os.path.exists(gt_path):
                gt = Image.open(gt_path).convert('RGB')
                gt_tensor = transform(gt).unsqueeze(0).to(device)
                rectified_tensor = rectified.unsqueeze(0)
                score = ssim_func(rectified_tensor, gt_tensor, data_range=1.0).item()
                ssim_scores.append(score)
                # print(f"Processed {base_name} | SSIM: {score:.4f}")
            # else:
                # print(f"Processed {base_name} | no ground truth found")
        # else:
        #     print(f"Processed {base_name}")

    if ssim_scores:
        print(f"\nMean SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")
        print(f"Min SSIM: {min(ssim_scores):.4f}")
        print(f"Max SSIM: {max(ssim_scores):.4f}")

if __name__ == '__main__':
    main()