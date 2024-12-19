import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor
import argparse
from pathlib import Path

matplotlib.use('TkAgg')

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def calculate_weighted_centroid(mask):
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    if mask_uint8.max() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        if num_labels > 1:
            largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            center_x, center_y = centroids[largest_component_idx]
            return int(center_x), int(center_y)
    return mask.shape[1] // 2, mask.shape[0] // 2

def visualize_results(frame_idx, out_obj_ids, out_mask_logits, prompts, video_dir, frame_names):
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    plt.imshow(img)
    
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        show_mask(mask, plt.gca(), obj_id=out_obj_id)
        center_x, center_y = calculate_weighted_centroid(mask)
        plt.text(center_x, center_y, str(out_obj_id), 
                color='white', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    
    plt.savefig(f"frame_{frame_idx}_results.png")
    plt.show()
    plt.close()

def initialize_inference(predictor, video_dir):
    inference_state = predictor.init_state(video_path=video_dir)
    return inference_state

def track_objects(inference_state, objects_to_track):
    prompts = {}
    for obj_id, (points, labels) in objects_to_track.items():
        prompts[obj_id] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
    return prompts, out_obj_ids, out_mask_logits

def propagate_and_visualize(predictor, inference_state, video_dir, frame_names, output_video, frame_stride, batch_size=10):
    video_segments = {}
    num_frames = len(frame_names)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    fig, ax = plt.subplots(figsize=(6, 4))

    def update(frame_idx):
        ax.clear()
        ax.set_title(f"frame {frame_idx}")

        img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
        ax.imshow(img)
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            show_mask(out_mask, ax, obj_id=out_obj_id)
            center_x, center_y = calculate_weighted_centroid(out_mask)
            ax.text(center_x, center_y, str(out_obj_id), 
                    color='white', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    ani = FuncAnimation(fig, update, frames=range(0, len(frame_names), frame_stride), repeat=False)
    ani.save(output_video, writer='ffmpeg', fps=30)

def extract_and_save_frames(video_path, output_dir, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    target_fps = 5
    frame_interval = int(30 / target_fps)

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f'{frame_count:05d}.jpg')
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
        
    
    cap.release()
    print(f"Frame interval: {frame_interval}")
    print(f"Extracted and saved {saved_count} frames at {target_fps} FPS")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_input', type=str, required=True, help='Path to the input video directory')
    parser.add_argument('--video_output', type=str, default='output_video.mp4', help='Path to the output video file')
    parser.add_argument('--frame_stride', type=int, default=1, help='Stride for visualizing frames')
    parser.add_argument('--ckpt', type=str, default="./checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument('--model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument('--frames_dir', type=str, default="./frames")
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.ckpt[:14])
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.ckpt):
        ckpt = args.ckpt[14:]
        download_url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt}"
        print(f"Checkpoint not found at {args.ckpt}")
        print(f"Downloading from {download_url}")
        command = f"wget -P {checkpoint_dir} {download_url}"
        os.system(command)
        print("\nDownload complete!")
    else:
        print(f"Found existing checkpoint at {args.ckpt}")
    

    model_cfg = args.model_cfg
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    predictor = build_sam2_video_predictor(model_cfg, args.ckpt, device=device)

    extract_and_save_frames(args.video_input, args.frames_dir)

    video_dir = args.frames_dir
    frame_names = sorted(
        [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    inference_state = initialize_inference(predictor, video_dir)

    objects_to_track = {
        1: (np.array([[134, 334], [207, 703], [370, 525]], dtype=np.float32), np.array([1, 1, 1], np.int32)),
        2: (np.array([[222, 372]], dtype=np.float32), np.array([1], np.int32)),
        3: (np.array([[312, 362]], dtype=np.float32), np.array([1], np.int32)),
        4: (np.array([[131, 428]], dtype=np.float32), np.array([1], np.int32)),
        5: (np.array([[93, 579]], dtype=np.float32), np.array([1], np.int32)),
        6: (np.array([[112, 697]], dtype=np.float32), np.array([1], np.int32)),
        7: (np.array([[340, 604]], dtype=np.float32), np.array([1], np.int32)),
        8: (np.array([[374, 439]], dtype=np.float32), np.array([1], np.int32)),
        9: (np.array([[293, 525]], dtype=np.float32), np.array([1], np.int32)),
        10: (np.array([[269, 441]], dtype=np.float32), np.array([1], np.int32)),
        11: (np.array([[76, 512]], dtype=np.float32), np.array([1], np.int32)),
        12: (np.array([[252, 619]], dtype=np.float32), np.array([1], np.int32)),
        13: (np.array([[215, 512]], dtype=np.float32), np.array([1], np.int32)),
        14: (np.array([[43, 394]], dtype=np.float32), np.array([1], np.int32)),
    }

    prompts, out_obj_ids, out_mask_logits = track_objects(inference_state, objects_to_track)
    # visualize_results(0, out_obj_ids, out_mask_logits, prompts, video_dir, frame_names)
    propagate_and_visualize(predictor, inference_state, video_dir, frame_names, args.video_output, args.frame_stride)
