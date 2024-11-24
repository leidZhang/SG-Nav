import os
import sys
import math
import json
from copy import deepcopy
from collections import Counter

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
from scenegraph_utils import (
    clear_line,
    find_modes,
    get_pose_matrix,
    llm,
    compute_clip_features,
    crop_image_and_mask,
    get_cfg,
    get_sam_segmentation_dense
)


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
        text = text[:-1] + '\n'
        for edge in edges:
            text = text + edge.relation + '/'
        text = text[:-1]
        return text


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
        # self.cfg = process_cfg(self.cfg)
        self.cfg = get_cfg()

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
            You find that the largest object in the picture is a whilte pipe
            Response: Pipe
        ''' # get the captions
        self.prompt_gpt = '''
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
            All the pairs of objects are provided line by line, and you also need to response to each
            pair one by one with the same order. Here are 2 examples:
            1.
            Input:
            chair and table
            monitor and desk
            Response:
            next to
            above
            2.
            Input:
            sofa and TV
            plant and chair
            Response:
            opposite to
            behind
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
        ''' # GPT give distance. For some reason, this prompt is never used since the related methon has been commented out.
        self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"  # annotated by someone
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.chat = LLaVA('liuhaotian/llava-v1.5-7b')

    def reset(self):
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = set()
        self.subgraphs = set()

    def update_observation(self, observations):
        print(f'update_observation {self.agent.navigate_steps}...')
        # depth 1
        image_rgb = observations['rgb'].copy()
        depth_array = observations['depth'].copy()
        pose_matrix = get_pose_matrix(observations, self.agent.map_size_cm)

        camera_matrix = self.agent.camera_matrix
        K = np.array([
            [camera_matrix.f, 0, camera_matrix.xc],
            [0, camera_matrix.f, camera_matrix.zc],
            [0, 0, 1]
        ])

        self.segment2d(image_rgb)
        self.mapping3d(image_rgb, depth_array, cam_K=K, pose=pose_matrix) # depth 3

        self.get_caption()
        self.update_node(self.agent.obj_goal)
        self.update_edge(self.agent.obj_goal)
        # self.create_subgraphs(self.agent.obj_goal)
        # clear_line() # temp disable

    # BUG: for some reason the width of the image changed
    def segment2d(self, image_rgb):
        print('    segement2d...')
        print('        sam_segmentation...')

        mask, xyxy, conf = get_sam_segmentation_dense(
            self.sam_variant, self.mask_generator, image_rgb)
        clear_line()
        detections = Detections(
            xyxy=xyxy, # bounding box
            confidence=conf, # confidence score
            class_id=np.zeros_like(conf).astype(int),
            mask=mask, # segmentation mask
        )
        with torch.no_grad():
            print('        clip_feature...')
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.classes, self.device
            ) # image crops, image features, text features
            clear_line()

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

        # clear_line() # temp disable

    def mapping3d(self, image_rgb, depth_array, cam_K, pose):
        depth_array = depth_array[..., 0]
        gobs = self.segment2d_results[-1] # graph of objects and background?
        unt_pose = pose
        adjusted_pose = unt_pose
        idx = len(self.segment2d_results) - 1

        # NOTE: requires open3d to be installed
        # Get fore ground objects
        fg_detection_list, _ = gobs_to_detection_list( # fg: foreground, bg: background
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
        if len(fg_detection_list) == 0:
            clear_line()
            return

        # The 1st step of mapping, so no need to merge objects
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
            # Skip the similarity computation
            self.objects_post = filter_objects(self.cfg, self.objects)
            clear_line()
            return

        # Merge objects with the existing map
        print('        compute_spatial_similarities...')
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        clear_line()
        print('        compute_visual_similarities...')
        visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
        clear_line()
        print('        aggregate_similarities...')
        agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
        clear_line()

        agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')

        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)

        self.objects_post = filter_objects(self.cfg, self.objects)
        clear_line()

    def get_caption(self):
        print('    get_caption...')
        llava_time = 0
        for idx, object in enumerate(self.objects_post):
            conf = object["conf"]
            conf = np.array(conf)
            idx_most_conf = np.argsort(conf)[::-1]

            low_confidences = []

            image_list = []
            caption_list = []
            confidences_list = []
            low_confidences_list = []
            mask_list = []  # New list for masks
            score_list = []

            idx_most_conf = idx_most_conf[:self.max_detections_per_object]
            for idx_det in idx_most_conf:
                if self.segment2d_results[object["image_idx"][idx_det]]['image_appear_efficiency'][object["mask_idx"][idx_det]] != '':
                    continue

                image = self.segment2d_results[object["image_idx"][idx_det]]["image_rgb"]
                xyxy = object["xyxy"][idx_det]
                class_id = object["class_id"][idx_det]
                mask = object["mask"][idx_det]

                padding = 10
                x1, y1, x2, y2 = xyxy
                image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
                image_crop_modified = image_crop  # No modification

                if 'captions' not in object:
                    object['captions'] = []
                _w, _h = image_crop.size
                if _w * _h < 70 * 70:
                    low_confidences.append(True)
                    score_list.append(0.5)
                    continue
                else:
                    low_confidences.append(False)

                self.chat.reset()
                print(f'        LLaVA {llava_time}...')
                llava_time = llava_time + 1
                # print(f"LLaVa prompt: {self.prompt_llava}")
                image_features = self.__get_image_features(image_crop_modified)
                caption = self.chat(
                    # image=image_crop_modified,
                    image_features=image_features,
                    query=self.prompt_llava
                )  # added by someone

                print("\n")
                caption = caption.replace('\n', '').replace('.', '').lower() # .replace(' ', '')
                caption = caption.split(' ')[-1]
                print(f"Caption given by LLaVa: {caption}\n")
                clear_line()
                object['captions'].append(caption)
                self.segment2d_results[object["image_idx"][idx_det]]['image_appear_efficiency'][object["mask_idx"][idx_det]] = 'done'

                conf_value = conf[idx_det]
                image_list.append(image_crop)
                caption_list.append(caption)
                confidences_list.append(conf_value)
                low_confidences_list.append(low_confidences[-1])
                mask_list.append(mask_crop)  # Add the cropped mask
        # clear_line()

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
            room_label = torch.where(self.agent.room_map[0, :, y, x]==1)[0]
            if room_label.numel() == 1:
                room_label = room_label.item()
                if node.room_node:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
        # score all the new nodes
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                caption = node.caption
                print(f'        LLM {i}/{len(self.nodes)}...')

                gpt_prompt = self.prompt_gpt.format(caption, obj_goal)
                # print("GPT Prompt", gpt_prompt)
                response = llm(prompt=gpt_prompt, llm_name=self.llm_name)
                response = caption + "is" + obj_goal # TODO: Delete this later
                print(f"Response by GPT: {response}\n")
                clear_line()
                node.reason = response
        # clear_line()

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
        prompt = self.prompt_edge_proposal + '{} and {}.\n' * len(new_edges)
        prompt = prompt.format(*node_pairs)
        # print(f"Edge proposal prompt to GPT: {prompt}")
        relations = llm(prompt=prompt, llm_name=self.llm_name)
        relations = relations.split('\n')
        relations = ["next to" for _ in range(len(new_edges))] # TODO: Delete this later
        print(f"Edge proposal {relations} is provided by GPT\n")

        if len(relations) == len(new_edges):
            for i, relation in enumerate(relations):
                new_edges[i].relation = relation
        # clear_line()
        # discriminate all relation proposals
        self.free_map = self.agent.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
        for i, new_edge in enumerate(new_edges):
            print(f'        discriminate_relation  {i}/{len(new_edges)}...')
            if new_edge.relation == None or not self.discriminate_relation(new_edge):
                new_edge.delete()
            clear_line()
        # get edges set
        self.edges = set()
        for node in self.nodes:
            self.edges.update(node.edges)
        clear_line()

    def create_subgraphs(self, obj_goal):
        print('    create_subgraphs...')
        self.subgraphs.clear()
        for node in self.nodes:
            self.subgraphs.add(SubGraph(node))
        for i, subgraph in enumerate(self.subgraphs):
            subgraph_text = subgraph.get_subgraph_2_text()
            print(f'        LLM {i}/{len(self.subgraphs)}...')

            prompt = self.prompt_score_subgraph.format(subgraph_text, obj_goal)
            print("Score subgraph prompt to GPT: ", prompt)
            response = llm(prompt=prompt, llm_name=self.llm_name)
            print(f"Score subgraph result {response} given by GPT\n")

            # clear_line()
            distance = response.split(' ')[0]
            try:
                distance = float(distance)
            except ValueError:
                distance = 2
            if distance < 0.1:
                distance = 0.1
            score = 1 / distance
            subgraph.distance = distance
            subgraph.score = score
        clear_line()

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
                x = x1 + (x2 - x1) * i / n
                y = y1 + (y2 - y1) * i / n
                if not self.free_map[y, x]:
                    return False
            return True

    def init_room_nodes(self):
        room_nodes = []
        for room_caption in self.agent.rooms:
            room_node = RoomNode(room_caption)
            room_nodes.append(room_node)
        return room_nodes

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

    # TODO: Temp method, please check the main function in llava_model.py
    def __get_image_features(self, image):
        image_tensor = self.chat.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        # print(f"image tensor", image_tensor.device)
        return self.chat.encode_image(
            image_tensor[None, ...].half().cuda()
        )


if __name__ == '__main__':
    scenegraph = SceneGraph()
    color_path = '/your/path/to/Replica/room0/results/frame000000.jpg'
    scenegraph.segment2d(color_path)
    a = 1