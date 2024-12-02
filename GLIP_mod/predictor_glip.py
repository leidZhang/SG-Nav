import cv2
import torch
import re
import numpy as np
from typing import List, Union
import nltk
import inflect
from transformers import AutoTokenizer
from torchvision import transforms as T

from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

engine = inflect.engine()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

import timeit


class GLIPDemo(object):
    def __init__(self,
                 cfg,
                 confidence_threshold=0.7,
                 min_image_size=None,
                 show_mask_heatmaps=False,
                 masks_per_dim=5,
                 ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each tokens
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.tokenizer = self.build_tokenizer()

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size) if self.min_image_size is not None else lambda x: x,
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
        return tokenizer

    def run_ner(self, caption):
        noun_phrases = find_noun_phrases(caption)
        if 'room' in caption:
            noun_phrases = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
        relevant_phrases = noun_phrases
        labels = noun_phrases
        self.entities = labels

        tokens_positive = []

        for entity, label in zip(relevant_phrases, labels):
            try:
                # search all occurrences and mark them as different entities
                for m in re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())

        return tokens_positive

    def inference(self, original_image, original_caption, thresh=None):
        predictions = self.compute_prediction(original_image, original_caption)
        if not thresh:
            top_predictions = self._post_process_fixed_thresh(predictions)
        else:
            top_predictions = self._post_process(predictions, thresh)
        return top_predictions

    def inference_batch(self, original_image_list, original_caption, thresh=None):
        predictions = self.compute_prediction_batch(original_image_list, original_caption)
        top_predictions_list = []
        for prediction in predictions:
            if not thresh:
                top_predictions_list.append(self._post_process_fixed_thresh(prediction))
            else:
                top_predictions_list.append(self._post_process(prediction, thresh))
        return top_predictions_list
    
    def run_on_web_image(self, original_image, original_caption, thresh=0.5):
        predictions = self.compute_prediction(original_image, original_caption)
        top_predictions = self._post_process(predictions, thresh)

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_entity_names(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)

        return result, top_predictions

    def compute_prediction(self, original_image, original_caption):
        # image
        # print('img shape: ',original_image.shape)
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        tokenized = self.tokenizer([original_caption], return_tensors="pt")
        tokens_positive = self.run_ner(original_caption)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        # print(height, width)

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction

    def compute_prediction_batch(self, original_image_list, original_caption):
        # image
        image_list = []
        for image in original_image_list:
            # print('img shape pano: ', image.shape)
            image_list.append(self.transforms(image))
        
        image_list = to_image_list(image_list, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        tokenized = self.tokenizer([original_caption], return_tensors="pt")
        tokens_positive = self.run_ner(original_caption)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption for i in range(len(original_image_list))], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        for i, prediction in enumerate(predictions):
            # reshape prediction (a BoxList) into the original image size
            height, width = original_image_list[i].shape[:-1]
            predictions[i] = prediction.resize((width, height))
            # print(width, height, prediction.bbox)

            if prediction.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = prediction.get_field("mask")
                # always single image is passed at a time
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

        return predictions

    def _post_process_fixed_thresh(self, predictions):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = self.confidence_threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = self.confidence_threshold[0]
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = (30 * (labels[:, None] - 1) + 1) * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        return image

    def overlay_scores(self, image, predictions):
        scores = predictions.get_field("scores")
        boxes = predictions.bbox

        for box, score in zip(boxes, scores):
            box = box.to(torch.int64)
            image = cv2.putText(image, '%.3f' % score,
                                (int(box[0]), int((box[1] + box[3]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                (255, 255, 255), 1)

        return image

    def overlay_entity_names(self, image, predictions, names=None):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        new_labels = []
        if self.entities and self.plus:
            for i in labels:
                if i <= len(self.entities):
                    new_labels.append(self.entities[i - self.plus])
                else:
                    new_labels.append('object')
            # labels = [self.entities[i - self.plus] for i in labels ]
        else:
            new_labels = ['object' for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, new_labels):
            x = box[0]
            y = box[1] + 20
            s = template.format(label, score)
            cv2.putText(
                image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        # import pdb
        # pdb.set_trace()
        # masks = masks > 0.1

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 2)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]

        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET), None


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def find_noun_phrases(caption: str) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()
