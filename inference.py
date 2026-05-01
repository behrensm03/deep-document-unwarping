import argparse
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from model import DocumentReconstructionModel
from uv_dewarp import dewarp_with_uv

def load_img(path, img_size):
    img = Image.open(path).convert('RGB').resize((img_size, img_size), Image.BILINEAR)
    rgb_raw = np.array(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor = (torch.from_numpy(rgb_raw).float() / 255.0).permute(2,0,1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), rgb_raw # tensor for the model, rgb_raw for the visualization and dewarping

def uv_to_png(pred_uv):
    H, W = pred_uv.shape[:2]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[:, :, 0] = (pred_uv[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    vis[:, :, 1] = (pred_uv[:, :, 1] * 255).clip(0, 255).astype(np.uint8)
    return vis

def main():
    parser = argparse.ArgumentParser(description="Document dewarping inference")
    parser.add_argument('--input_dir',  required=True,            help='Folder of input images')
    parser.add_argument('--output_dir', default='./results',      help='Where to save outputs')
    parser.add_argument('--weights',    default='best_model.pth', help='Path to model weights')
    parser.add_argument('--img_size',   type=int, default=256,    help='Img size (must match training size)')
    args = parser.parse_args()

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = DocumentReconstructionModel().to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {args.weights}")

    # collect images from input dir
    image_files = sorted([
        f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']
    ])
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return
    print(f"Found {len(image_files)} images")

    # process the images
    with torch.no_grad():
        for n, frame in enumerate(tqdm(image_files, desc="Processing")):
            img_path = os.path.join(args.input_dir, frame)
            try:
                tensor, rgb_raw = load_img(img_path, args.img_size)
                tensor = tensor.to(device)

                # now run it through the model
                pred_uv_t = model(tensor)
                pred_uv = pred_uv_t[0].cpu().numpy().transpose(1,2,0).clip(0,1)  # (2, H, W) -> (H, W, 2)

                # call dewarping function
                rectified = dewarp_with_uv(rgb_raw, pred_uv, out_size=args.img_size, mask=None)

                # save outputs
                rect_path = os.path.join(args.output_dir, f"rectified_{n}.png")
                uv_path = os.path.join(args.output_dir, f"predicted_uv_{n}.png")
                Image.fromarray(rectified).save(rect_path)
                Image.fromarray(uv_to_png(pred_uv)).save(uv_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Processing complete. Results saved to {args.output_dir}")
                

if __name__ == "__main__":
    main()
