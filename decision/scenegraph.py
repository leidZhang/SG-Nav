import os
import sys
import math
import json
import random
from datetime import datetime
from copy import deepcopy

sys.path.append('/home/exx/Desktop/embodied_ai/concept-graphs')
import cv2
import numpy as np
import torch
import open_clip
from openai import OpenAI
from supervision import (
    Detections,
    BoxAnnotator,
    MaskAnnotator,
)
from PIL import Image

import dataclasses
import omegaconf
from omegaconf import DictConfig
from pathlib import PosixPath
from pathlib import Path
from supervision.draw.color import ColorPalette
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from conceptgraph.llava.llava_model import LLaVaChat as LLaVA
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import (
    filter_objects,
    gobs_to_detection_list,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)
from .scenegraph_utils import *


class RoomNode():
    def __init__(self, room_caption):
        self.room_caption = room_caption
        self.nodes = set()


class Node():
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def update_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True

        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.edges.clear()


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = [self.node1.caption, self.node2.caption, self.relation]
        return text


class SubGraph():
    def __init__(self, center_node):
        self.center_node = center_node
        self.edges = self.center_node.edges
        self.center = self.center_node.center
        self.nodes = set()
        for edge in self.edges:
            self.nodes.add(edge.node1)
            self.nodes.add(edge.node2)

    def get_subgraph_2_text(self):
        text = ''
        edges = set()
        for node in self.nodes:
            text = text + node.caption + '/'
            edges.update(node.edges)
        # text = text[:-1] + '\n'
        node_list, text = text[:-1].split('/'), ''
        for edge in edges:
            text = text + edge.relation + '/'
        # text = text[:-1]
        edge_list = text[:-1].split('/')
        return node_list, edge_list


class SceneGraph():
    def __init__(self, agent, is_navigation=True, llm_name='GPT') -> None:
        self.agent = agent
        self.GSA_PATH = os.environ["GSA_PATH"]
        self.SAM_ENCODER_VERSION = "vit_h"
        self.SAM_CHECKPOINT_PATH = os.path.join(self.GSA_PATH, "./sam_vit_h_4b8939.pth")
        self.sam_variant = 'sam'
        self.device = 'cuda'
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.objects = MapObjectList(device=self.device)
        self.objects = MapObjectList(device=self.device)
        self.nodes = set()
        self.subgraphs = set()
        self.room_nodes = self.init_room_nodes()
        self.is_navigation = is_navigation
        self.llm_name = llm_name
        # self.cfg = get_cfg()
        # self.cfg = self.process_cfg(self.cfg)
        self.cfg = self.get_cfg()
        
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.reason = ''
        self.prompt_llava = '''
            You are an AI assistant witho commonsense and strong ability to identify the
            largest object in an indoor scene. 

            You need to provide the name of the largest object in the picture. Here is 2 example:
            1.
            You find that the largest object in the picture is a dog
            Response: Dog 
            2. 
            You find that the largest object in the picture is a white pipe
            Response: Pipe
        ''' # get the captions 
        self.prompt_gpt = '''
            You are an AI assistant with commonsense and strong ability to infer the
            distance between two objects in an indoor scene.
            You need to predict the most likely distance of two objects in a room.
            You need to answer the distance in meters and give your reason. Here is
            the JSON format:
            Input:
            {{"object1": "table", "object2": "chair"}}
            Response:
            {{"distance": 0.50, "reason": "Because there is always a chair next to the table."}}
            Now predict the distance and give your reason: {{"object1": {}, "object2": {}}}
        '''
        '''
            You are an AI assistant with commonsense and strong ability to compare the object and goal,
            then give the category relationship between them.
            You need to compare two given objects and give the category relationship between them.
            The objects are provided in the JSON format, and you need to provide the category relationship
            between them. Here are 2 examples:
            1.
            Input:
            {{"object": "chair", "goal": "table"}}
            Response:
            A chair is not a table
            2.
            Input:
            {{"object": "american flag", "goal": "flag"}}
            Response:
            An American flag is flag
            Now you predict the category relationship between these two objects:
            {{"object": {}, "goal": {}}}
        ''' # GPT give the confidence reason? whether the detected object is the object goal
        self.prompt_edge_proposal = '''
            You are an AI assistant with commonsense and strong ability to infer the spatical
            relationships in a indoor scene.
            You need to provide a spatial relationship between the several pairs of objects. Relationships
            include "next to", "above", "opposite to", "below", "inside", "behind", "in front of"
            All the pairs of objects are provided in JSON fromat, and you need to provide a JSON result that
            response to each pair in the format of "<number>": "<relationship>" with the same order. Here are 2 examples:
            1.
            Input:
            {"0": {"object1": "chair", "object2": "table"}, "1": {"object1": "monitor", "object2": "desk"}
            Response:
            {"0": "next to", "1": "above"}
            2.
            Input:
            {"0": {"object1": "sofa", "object2": "TV"}, "1": {"object1": "plant", "object2": "chair"}}
            Response:
            {"0": "opposite to", "1": "behind"}
            Now you predict the spatial relationship between these pairs of objects:
        ''' # GPT give the relationship based on the given object nodes' caption.

        self.prompt_discriminate_relation = '''
            You are an AI assistant with commonsense and strong ability to judge whether the
            spatial relationship between the objects is correct or not.
            You will be provided with an image consisting of two objects, the text input of their caption and their
            spatial relationship in JSON format. Based on this information, you need to judge whether the relationship
            is correct or not. If the relationship is correct, answer "Yes", otherwise, answer "No". Here are 2 examples:
            1.
            You find that there is a sofa next to a TV from the image.
            Input text: {{"object1": sofa, "object2": TV, "relation": "next to"}}.
            Response: Yes
            2.
            You find that there is a cat above the table from the image.
            Input text: {{"object1": cat, "object2": table, "relation": "below"}}.
            Response: No
            Now with the provided image and the text: {{"object1": {}, "object2": {}, "relation": {}}}, is this
            spatial relationship correct?
        ''' # Does this relation make sense? LLaVa will answer this question.
        self.prompt_score_subgraph = '''
            You are an AI assistant with commonsense and strong ability to infer the distance between a subgraph
            and a goal in an indoor scene. 

            You need to predict the most likely distance of a subgraph and a goal in a room. You need to answer the distance in 
            meters and give your reason in the format of {{"distance": <number>, "reason": <reason_string>}}. Here is the JSON format:
            Input:
            {{"subgraph": {{"nodes": ["sofa", "table", ...], "edges": ["sofa next to table", ...]}}, "goal": "TV"}}
            Response: 
            {{"distance": 2.00, "reason": "Becasue TV and sofa are on both sides of table."}}

            Now predict the distance and give your reason: 
            {{"subgraph": {}, "goal": {}}}
        ''' # GPT give distance. For some reason, this prompt is never used since the related methon has been commented out.
        self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"  # annotated by someone
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.chat = LLaVA('liuhaotian/llava-v1.5-7b')

    def update_observation(self, observations):
        print(f'update_observation {self.agent.navigate_steps}...')
        image_rgb = observations['rgb'].copy()
        depth_array = observations['depth'].copy()
        pose_matrix = self.get_pose_matrix(observations, self.agent.map_size_cm)

        camera_matrix = self.agent.camera_matrix
        K = np.array([
            [camera_matrix.f, 0, camera_matrix.xc],
            [0, camera_matrix.f, camera_matrix.zc],
            [0, 0, 1]
        ])

        self.segment2d(image_rgb)
        self.mapping3d(image_rgb, depth_array, cam_K=K, pose=pose_matrix) 
        self.get_caption() 
        self.update_node(self.agent.obj_goal)
        self.update_edge(self.agent.obj_goal)
        self.create_subgraphs(self.agent.obj_goal)
        # self.clear_line() # temp disable

    # BUG: for some reasone the width of the image changed
    def segment2d(self, image_rgb):
        print('    segement2d...')
        print('        sam_segmentation...')

        mask, xyxy, conf = self.get_sam_segmentation_dense(
            self.sam_variant, self.mask_generator, image_rgb)
        self.clear_line()
        detections = Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros_like(conf).astype(int),
            mask=mask,
        )
        with torch.no_grad():
            print('        clip_feature...')
            image_crops, image_feats, text_feats = self.compute_clip_features(
                image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.classes, self.device)
            self.clear_line()

        image_appear_efficiency = [''] * len(image_crops)
        self.segment2d_results.append({
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": self.classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "image_appear_efficiency": image_appear_efficiency,
            "image_rgb": image_rgb,
        })
        
        # self.clear_line() # temp disable

    def mapping3d(self, image_rgb, depth_array, cam_K, pose):
        # depth_array = depth_array[..., 0]
        depth_array = depth_array.reshape(480, 640)

        gobs = None # stands for grounded SAM observations

        gobs = self.segment2d_results[-1]
        
        unt_pose = pose
        
        adjusted_pose = unt_pose
            
        idx = len(self.segment2d_results) - 1
        print(depth_array.shape)
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            # is_navigation = self.is_navigation
            # color_path = color_path,
        ) # BUG: The image_rgb changed after this step

        # TODO: For some reason I have to use this line, and no idea of the concequence
        self.segment2d_results[-1]["image_rgb"] = image_rgb         
        # print(f"Image size after gob {image_rgb.shape}")
        # print(f"Image size after gob {self.segment2d_results[0]['image_rgb'].shape}")
            
        if len(fg_detection_list) == 0:
            self.clear_line()
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            self.clear_line()
            return
                
        print('        compute_spatial_similarities...')
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        self.clear_line()
        print('        compute_visual_similarities...')
        visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
        self.clear_line()
        print('        aggregate_similarities...')
        agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
        self.clear_line()
        
        agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim) 
        self.objects_post = filter_objects(self.cfg, self.objects)
        self.clear_line()
            
    def get_caption(self):
        print('    get_caption...')
        llava_time = 0
        for idx, object in enumerate(self.objects_post):
            conf = object["conf"]
            conf = np.array(conf)
            idx_most_conf = np.argsort(conf)[::-1]

            features = []
            captions = []
            low_confidences = []
            
            image_list = []
            caption_list = []
            confidences_list = []
            low_confidences_list = []
            mask_list = []  # New list for masks
            score_list = []

            # print(f"idx most conf {idx_most_conf}")
            idx_most_conf = idx_most_conf[:self.max_detections_per_object]

            for idx_det in idx_most_conf:
                if self.segment2d_results[object["image_idx"][idx_det]]['image_appear_efficiency'][object["mask_idx"][idx_det]] == '':
                    image = self.segment2d_results[object["image_idx"][idx_det]]["image_rgb"]
                    xyxy = object["xyxy"][idx_det]
                    class_id = object["class_id"][idx_det]
                    mask = object["mask"][idx_det]

                    padding = 10
                    x1, y1, x2, y2 = xyxy
                    image_crop, mask_crop = self.crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
                    image_crop_modified = cv2.cvtColor(np.array(image_crop), cv2.COLOR_BGR2RGB) # image_crop # No modification
                    if 'captions' not in object:
                        object['captions'] = {}
                    _w, _h = image_crop.size
                    if _w * _h < 70 * 70: # or _w / _h >= 5 or _h / _w >= 5:
                        low_confidences.append(True)
                        score_list.append(0.5)
                        continue
                    else:
                        low_confidences.append(False)


                    self.chat.reset()
                    print(f'        LLaVA {llava_time}...')
                    llava_time = llava_time + 1

                    image_features = self.__get_image_features(image_crop_modified)
                    image_features.to("cuda:0")

                    caption = self.chat(
                        # image=image_crop_modified, 
                        image_features=image_features,
                        query=self.prompt_llava
                    )  # added by someone

                    caption = caption.replace('\n', '').replace('.', '').lower() # .replace(' ', '')
                    expression = caption.split(' ')
                    
                    if len(expression) > 1:
                        caption = caption.split(' ')[-1]
                    # cv2.imwrite(f"{object['image_idx'][idx_det]}.jpg", np.array(image_crop_modified)) 
                    # print(f"Caption given by LLaVa: {caption}")            
       
                    self.clear_line()
                    object['captions'][caption] = 1 if caption not in object['captions'].keys() else object['captions'][caption] + 1
                    self.segment2d_results[object["image_idx"][idx_det]]['image_appear_efficiency'][object["mask_idx"][idx_det]] = 'done'
                        
                    conf_value = conf[idx_det]
                    image_list.append(image_crop)
                    caption_list.append(caption)
                    confidences_list.append(conf_value)
                    low_confidences_list.append(low_confidences[-1])
                    mask_list.append(mask_crop)  # Add the cropped mask

        self.clear_line()
        for i, obj in enumerate(self.objects_post):
            print(f"{i}: {obj['captions']}") # {obj['xy']}
        # self.clear_line()

    def update_node(self, obj_goal):
        print('    update_node...')
        node_num_ori = len(self.nodes)
        node_num_new = len(self.objects_post)
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            caption_new = find_modes(self.objects_post[i]['captions'])[0]
            if caption_ori != caption_new:
                node.update_caption(caption_new)
        # add new nodes
        for i in range(node_num_ori, node_num_new):
            new_node = Node()
            caption = find_modes(self.objects_post[i]['captions'])[0]
            new_node.update_caption(caption)
            new_node.object = self.objects_post[i]
            self.nodes.add(new_node)
        # get node.center and node.room
        for node in self.nodes:
            points = node.object['pcd'].points
            points = np.asarray(points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.agent.resolution)
            y = int(center[1] * 100 / self.agent.resolution)
            y = self.agent.map_size - 1 - y
            node.center = [x, y]

            if abs(x) <= 800 and abs(y) <= 800: # Avoid index error
                room_label = torch.where(self.agent.room_map[0, :, y, x]==1)[0]
                if room_label.numel() == 1:
                    room_label = room_label.item()
                    if node.room_node:
                        node.room_node.nodes.discard(node)
                    node.room_node = self.room_nodes[room_label]
                    node.room_node.nodes.add(node)                
            else:
                print(node.center, "is out of the map")


        # score all the new nodes
        for i, node in enumerate(self.nodes):
            if not node.is_new_node:
                continue

            caption = node.caption
            print(f'        LLM {i}/{len(self.nodes)}...')
            try:
                gpt_prompt = self.prompt_gpt.format(caption, obj_goal)
                distance, reason = get_subgraph_score_from_llm(gpt_prompt, "GPT")
                distance = distance.split(' ')[0]
                distance = float(distance)
            except Exception:
                distance = 2.0
                reason = ""
            if distance < 0.1:
                distance = 0.1

            node.score = 1 / distance
            # node.reason = reason                
            # print(f"Distance between {node.caption} and {obj_goal} is {distance}")

            # print("GPT Prompt", gpt_prompt)
            # response = llm(prompt=gpt_prompt, llm_name=self.llm_name)
            # print(f"Response by GPT: {response}")
            # self.clear_line()
        # self.clear_line()

    def update_edge(self, obj_goal):
        print('    update_edge...')
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        print(f"New nodes {len(new_nodes)}, old nodes {len(old_nodes)}")
        # create the edge between new_node and old_node
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                new_node.edges.add(new_edge)
                old_node.edges.add(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                new_node1.edges.add(new_edge)
                new_node2.edges.add(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        print(f'        LLM get all relation proposals...')
        node_pairs = []
        for new_edge in new_edges:
            node_pairs.append(new_edge.node1.caption)
            node_pairs.append(new_edge.node2.caption)
        if len(node_pairs) == 0:
            print("No node pair detected!")
            return

        # Potential bug: GPT may provide some wrong format!     
        relations = get_relations_from_llm(node_pairs, self.prompt_edge_proposal, self.llm_name)

        print(f"Edge proposal {relations}")
        print(f"Relations: {len(relations)}, new edges {len(new_edges)}")
        self.free_map = self.agent.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5        
        if len(relations) == len(new_edges):
            for i, relation in enumerate(relations):
                new_edges[i].relation = relation
        # discriminate all relation proposals

        for i, new_edge in enumerate(new_edges): 
            print(f'        discriminate_relation  {i}/{len(new_edges)}...')
            # print(f"New edge relation: {new_edge.relation}")
            if new_edge.relation == None or not self.discriminate_relation(new_edge):
                new_edge.delete()
            self.clear_line()
        # get edges set
        self.edges = set()

        for node in self.nodes:
            self.edges.update(node.edges)
        # self.clear_line()

    # BUG: Sometimes the GPT will give wrong format!
    def create_subgraphs(self, obj_goal):
        print('    create_subgraphs...')
        self.subgraphs.clear() 
        print(f"Num of node: {len(self.nodes)}")
        for node in self.nodes:
            self.subgraphs.add(SubGraph(node))
        for i, subgraph in enumerate(self.subgraphs):
            subgraph_text = subgraph.get_subgraph_2_text()
            # print(subgraph_text[0], subgraph_text[1])
            if subgraph_text[0][0] == '' and subgraph_text[1][0] == '':
                subgraph.score = 0.5 # Nothing in the subgraph
                continue 
            
            print(f'        LLM {i}/{len(self.subgraphs)}...')
            # print("Score subgraph prompt to GPT: ", prompt)
            # response = llm(prompt=prompt, llm_name=self.llm_name)

            # self.clear_line()
            # distance = response.split(' ')[0]
            try:
                prompt = self.prompt_score_subgraph.format(subgraph_text, obj_goal)                
                distance, reason = get_subgraph_score_from_llm(prompt, self.llm_name)
                # print(f"Score subgraph result: {distance}, since {reason} given by GPT\n")             
                distance = float(distance.split(' ')[0])
            except json.JSONDecodeError:
                print("GPT gave wrong format")
                distance = 2
            except ValueError:
                print("GPT gave wrong format")
                distance = 2
            if distance < 0.1:
                distance = 0.1
            score = 1 / distance
            subgraph.distance = distance
            subgraph.score = score
        self.clear_line()

    def init_room_nodes(self):
        room_nodes = []
        for room_caption in self.agent.rooms:
            room_node = RoomNode(room_caption)
            room_nodes.append(room_node)
        return room_nodes

    def get_cfg(self):
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
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        return cfg

    def get_sam_mask_generator(self, variant:str, device) -> SamAutomaticMaskGenerator:
        if variant == "sam":
            sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
            sam.to(device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=12,
                points_per_batch=144,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                min_mask_region_area=100,
            )
            return mask_generator
        elif variant == "fastsam":
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def get_sam_segmentation_dense(
        self, variant:str, model, image: np.ndarray
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

    def compute_clip_features(self, image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
        backup_image = image.copy()
        
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
            self.clear_line()
            
            print('            encode_text...')
            class_id = detections.class_id[idx]
            tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            self.clear_line()
            
            crop_feat = crop_feat.cpu().numpy()
            text_feat = text_feat.cpu().numpy()

            image_crops.append(cropped_image)
            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)

        return image_crops, image_feats, text_feats

    def vis_result_fast(
        self,
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

    def process_cfg(self, cfg: DictConfig):
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

    def crop_image_and_mask(self, image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
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
    
    def get_pose_matrix(self, observations, map_size_cm):
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

    def get_scenegraph_object_list(self):
        scenegraph_object_list = []
        for node in self.nodes:
            center = node.center
            score = node.score
            scenegraph_object_list.append({'center': center, 'score': score})
            # scenegraph_object_list.append(node.caption)
        return scenegraph_object_list

    def get_scenegraph_subgraph_list(self):
        scenegraph_subgraph_list = []
        for subgraph in self.subgraphs:
            center = subgraph.center
            score = subgraph.score
            scenegraph_subgraph_list.append({'center': center, 'score': score})
        return scenegraph_subgraph_list
    
    def get_scene_graph_text(self):
        scene_graph_text = {'nodes': [], 'edges': []}
        for node in self.nodes:
            scene_graph_text['nodes'].append(node.caption)
        for edge in self.edges:
            scene_graph_text['edges'].append(edge.text())
        scene_graph_text = json.dumps(scene_graph_text)
        return scene_graph_text
    
    def get_reason_text(self):
        sorted_nodes = sorted(list(self.nodes))
        reason_num = min(len(sorted_nodes), 4)
        reason_text = []
        for i in range(reason_num):
            reason_text.append(sorted_nodes[i].reason)
        reason_text = json.dumps(reason_text)
        return reason_text
    
    def visualize_objects(self):
        points_all = []
        for object in self.objects_post:
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
            
    def clear_line(self):  
        sys.stdout.write('\033[F')
        sys.stdout.write('\033[J')
        sys.stdout.flush()  
    
    # TODO: Temp method, please check the main function in llava_model.py
    def __get_image_features(self, image):
        image_tensor = self.chat.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        # print(f"image tensor", image_tensor.device)
        return self.chat.encode_image(
            image_tensor[None, ...].half().cuda()
        )
        
    def discriminate_relation(self, edge):
        image_idx1 = edge.node1.object["image_idx"]
        image_idx2 = edge.node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = edge.node1.object["conf"][image_idx1.index(idx)]
            conf2 = edge.node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        # discriminate short edge
        if len(image_idx) > 0:
            image = self.segment2d_results[idx_max]["image_rgb"]
            image = Image.fromarray(image)
            self.chat.reset()

            image_features = self.__get_image_features(image)
            print("Device i: ", image_features.device, "\n")

            prompt = self.prompt_discriminate_relation.format(
                edge.node1.caption, 
                edge.node2.caption, 
                edge.relation
            )
            # print(f"LLaVa relation prompt: {prompt}")
            response = self.chat(
                # image=image, 
                image_features=image_features,
                query=prompt
            )  # added by someone
            print(f"LLaVa response in discriminate relation: {response}\n")
            
            if 'yes' in response.lower():
                return True
            else:
                return False
        # discriminate long edge
        else:
            # discriminate same room 
            if edge.node1.room_node != edge.node2.room_node:
                return False
            x1, y1 = edge.node1.center
            x2, y2 = edge.node2.center
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > self.agent.map_size // 20:
                return False
            alpha = math.atan2(y2 - y1, x2 - x1)  
            sin_2alpha = 2 * math.sin(alpha) * math.cos(alpha)
            if not -0.05 < sin_2alpha < 0.05:
                return False
            # discriminate occlusion
            n = 8
            for i in range(1, n):
                x = int(x1 + (x2 - x1) * i / n)
                y = int(y1 + (y2 - y1) * i / n)
                if not self.free_map[y, x]:
                    return False
            return True
        
    def verify_goal(self, goal_xy):
        goal_x, goal_y = goal_xy
        distances = []
        subgraphs_list = list(self.subgraphs)
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
    
    def reset(self):
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = set()
        self.subgraphs = set()

if __name__ == '__main__':
    scenegraph = SceneGraph()
    color_path = '/your/path/to/Replica/room0/results/frame000000.jpg'
    scenegraph.segment2d(color_path)
    a = 1