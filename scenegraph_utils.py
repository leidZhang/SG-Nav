import os
import sys
import math
import json
import dataclasses
from collections import Counter

import cv2
import torch
import numpy as np
from PIL import Image
from openai import OpenAI
import omegaconf
from omegaconf import DictConfig
from supervision.draw.color import ColorPalette
from supervision import (
    Detections,
    BoxAnnotator,
    MaskAnnotator,
)
from pathlib import PosixPath
from pathlib import Path


def get_pose_matrix(observations, map_size_cm):
    x = map_size_cm / 100.0 / 2.0 + observations['gps'][0]
    y = map_size_cm / 100.0 / 2.0 - observations['gps'][1]
    t = (observations['compass'] - np.pi / 2)[0] # input degrees and meters
    pose_matrix = np.array([
        [np.cos(t), -np.sin(t), 0, x],
        [np.sin(t), np.cos(t), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return pose_matrix


def llm(prompt, llm_name='GPT'):
    if llm_name == 'GPT':
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chat_completion = client.chat.completions.create(  # added by someone
                model="gpt-3.5-turbo",
                # model="gpt-4",  # gpt-4
                messages=[{"role": "user", "content": prompt}],
                # timeout=10,  # Timeout in seconds
            )
            return chat_completion.choices[0].message.content
        except:
            return ''


def verify_goal(subgraphs, goal_xy):
    goal_x, goal_y = goal_xy
    distances = []
    subgraphs_list = list(subgraphs)
    for subgraph in subgraphs_list:
        center_x, center_y = subgraph.center
        distance = math.sqrt((goal_x - center_x)**2 + (goal_y - center_y)**2)
        distances.append(distance)
    distances = torch.tensor(distances).cuda()
    sorted_distances, indices = torch.sort(distances, descending=True)
    scores = [subgraphs_list[indices[i]].score for i in range(3)]
    score = sum(scores) / len(scores)
    score_threshold = 0.2
    return score > score_threshold


def find_modes(lst):
    if len(lst) == 0:
        return ['object']
    else:
        counts = Counter(lst)
        max_count = max(counts.values())
        modes = [item for item, count in counts.items() if count == max_count]
        return modes


def get_scenegraph_object_list(nodes):
    scenegraph_object_list = []
    for node in nodes:
        center = node.center
        score = node.score
        scenegraph_object_list.append({'center': center, 'score': score})
        # scenegraph_object_list.append(node.caption)
    return scenegraph_object_list


def get_scenegraph_subgraph_list(subgraphs):
    scenegraph_subgraph_list = []
    for subgraph in subgraphs:
        center = subgraph.center
        score = subgraph.score
        scenegraph_subgraph_list.append({'center': center, 'score': score})
    return scenegraph_subgraph_list


def get_scene_graph_text(nodes, edges):
    scene_graph_text = {'nodes': [], 'edges': []}
    for node in nodes:
        scene_graph_text['nodes'].append(node.caption)
    for edge in edges:
        scene_graph_text['edges'].append(edge.text())
    scene_graph_text = json.dumps(scene_graph_text)
    return scene_graph_text


def get_reason_text(nodes):
    sorted_nodes = sorted(list(nodes))
    reason_num = min(len(sorted_nodes), 4)
    reason_text = []
    for i in range(reason_num):
        reason_text.append(sorted_nodes[i].reason)
    reason_text = json.dumps(reason_text)
    return reason_text


def visualize_objects(objects_post):
    points_all = []
    for object in objects_post:
        points = object['pcd'].points
        points = np.asarray(points)
        colors = np.zeros_like(points, dtype=np.int64)
        colors[:, 0] = 0
        colors[:, 1] = 0
        colors[:, 2] = 0
        points = np.concatenate([points, colors], axis=1)
        points_all.append(points)
    points_all = np.concatenate(points_all, axis=0)
    np.savetxt('', points_all)
    return


def clear_line():
    sys.stdout.write('\033[F')
    sys.stdout.write('\033[J')
    sys.stdout.flush()


def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    image = Image.fromarray(image)

    padding = 20  # Adjust the padding amount as needed

    image_crops = []
    image_feats = []
    text_feats = []

    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        # Get the preprocessed image for clip from the crop
        print('            encode_image...')
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        clear_line()

        print('            encode_text...')
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        clear_line()

        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)

    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape[:2], mask.shape))
        return None, None

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None

    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop, mask_crop


def vis_result_fast(
    image: np.ndarray,
    detections: Detections,
    classes: list,
    color = ColorPalette.DEFAULT,
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results.
    This is fast but of the same resolution of the input image, thus can be blurry.
    '''
    # annotate image with detections
    box_annotator = BoxAnnotator(
        color = color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = MaskAnnotator(
        color = color
    )
    labels = [f"{classes[class_id]} {confidence:0.2f}" for confidence, class_id in zip(detections.confidence, detections.class_id)]  # added by someone

    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels

def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg


def get_cfg(is_navigation):
    cfg = {
        'dataset_root': PosixPath('/your/path/to/Replica'),
        'dataset_config': PosixPath('/your/path/to/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml'),
        'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680,
        'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}',
        'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda',
        'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5,
        'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95,
        'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5,
        'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True,
        'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7,
        'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1,
        'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub',
        'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True,
        'render_camera_path': 'replica_room0.json', 'max_num_points': 512
    }
    cfg = DictConfig(cfg)
    if is_navigation:
        cfg.sim_threshold = 0.8
        cfg.sim_threshold_spatial = 0.01
    return cfg


def get_sam_segmentation_dense(
    variant:str, model, image: np.ndarray
) -> tuple:
    '''
    The SAM based on automatic mask generation, without bbox prompting

    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]

    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
    if variant == "sam":
        results = model.generate(image)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    elif variant == "fastsam":
        # The arguments are directly copied from the GSA repo
        results = model(
            image,
            imgsz=1024,
            device="cuda",
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        raise NotImplementedError
    else:
        raise NotImplementedError