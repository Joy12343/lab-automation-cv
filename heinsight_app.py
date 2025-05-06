import asyncio
import datetime
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

import threading
from itertools import chain
from random import randint
import torch
from torchvision.ops import box_iou
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS


class HeinSight:
    NUM_ROWS = -1  # number of rows that the vial is split into. -1 means each individual pixel row
    VISUALIZE = False  # watch in real time, if False it will only make a video without showing it
    INCLUDE_BB = True  # show bounding boxes in output video
    READ_EVERY = 5  # only considers every 'READ_EVERY' frame -> faster rendering
    UPDATE_EVERY = 5  # classifies vial contents ever 'UPDATE_EVERY' considered frame
    LIQUID_CONTENT = ["Homo", "Hetero"]
    CAP_RATIO = 0.3  # this is the cap ratio of a HPLC vial
    NMS_RULES = {
        ("Homo", "Hetero"): 0.2,        # Suppress lower confidence when Homo overlaps with Hetero
        ("Hetero", "Residue"): 0.2,     # Suppress lower confidence when Hetero overlaps with Residue
        ("Solid", "Residue"): 0.2,      # Suppress lower confidence when Solid overlaps with Residue
        ("Empty", "Residue"): 0.2,      # Suppress lower confidence when Empty overlaps with Residue
    }

    def __init__(self, vial_model_path, contents_model_path):
        """
        Initialize the HeinSight system.
        """
        self._thread = None
        self._running = True
        self.vial_model = YOLO(vial_model_path)
        self.contents_model = YOLO(contents_model_path)
        self.vial_location = None
        self.vial_size = [80, 200]
        self.content_info = None
        self.color_palette = self._register_colors([self.contents_model])
        self.x_time = []
        self.turbidity_2d = []
        self.average_colors = []
        self.average_turbidity = []
        self.output = []
        self.output_dataframe = pd.DataFrame()
        self.output_frame = None
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6), height_ratios=[2, 1], constrained_layout=True)
        self._set_axes()

    def draw_bounding_boxes(self, image, bboxes, class_names, vial_location, phase_data = None, liquid_boxes = None, thickness=None, text_right: bool = False):
        """Draws rectangles on the input image."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = image.copy()

        vial_x1, vial_y1, vial_x2, vial_y2 = vial_location
        vial_y1 = int(self.CAP_RATIO * (vial_y2 - vial_y1)) + vial_y1  # Adjusted y1

        # change thickness and font scale based on the vial frame height
        thickness = thickness or max(1, int(self.vial_size[1] / 200))
        margin = thickness * 2
        text_thickness = max(1, min(int((thickness + 1) / 2), 3))  # (1, 3)
        fontscale = min(0.5 * thickness, 2)  # (0.5, 2)

        for i, rect in enumerate(bboxes):
            class_name = class_names[rect[-1]]
            color = self.color_palette.get(class_name)
            x1, y1, x2, y2 = [int(x) for x in rect[:4]]
            print(f"shape of image: {image.shape[0]}, {image.shape[1]}")
            print(f"Initial loc: {class_name} at ({x1, y1}), ({x2}, {y2})")
            x1 += vial_x1
            x2 += vial_x1
            y1 += vial_y1
            y2 += vial_y1
            print(f"vial size: ({vial_x1}, {vial_y1}), ({vial_x2}, {vial_y2})")
            print(f"Drawing box: {class_name} at ({x1}, {y1}), ({x2}, {y2})")
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontscale,
                                                                  text_thickness)
            text_location = (
                x2 - text_width - margin if text_right ^ (class_name == "Solid") else x1 + margin,
                y1 + text_height + margin
            )
            cv2.putText(output_image, class_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, fontscale, color,
                        text_thickness)

        if liquid_boxes and phase_data:
            for i, rect in enumerate(liquid_boxes):
                x1, y1, x2, y2 = [int(x) for x in rect[:4]]
                x1 += vial_x1
                x2 += vial_x1
                y1 += vial_y1
                y2 += vial_y1
                print(f"draw bounding boxes iterating at {i}")
                if phase_data and f'volume_{i+1}' in phase_data:
                    print(f"drawing volume_{i+1}") 
                    small_fontscale = fontscale * 0.6
                    volume_text = f"{phase_data[f'volume_{i+1}']:.2f} mL"
                    (v_width, v_height), _ = cv2.getTextSize(volume_text, cv2.FONT_HERSHEY_SIMPLEX, small_fontscale,
                                                                  text_thickness)
                    volume_location = (x2 - margin - v_width, y1 + margin + text_height)  # Upper-right of the bounding box
                    color = (255, 255, 255)
                    cv2.putText(output_image, volume_text, volume_location, cv2.FONT_HERSHEY_SIMPLEX, small_fontscale, color, text_thickness)

        return output_image

    def find_vial(self, frame, ):
        """
        Detect the vial in video frame with YOLOv8
        :param frame: raw input frame
        :return result: np.ndarray or None: Detected vial bounding box or None if no vial is found.
        """
        self.vial_model.conf = 0.5
        self.vial_model.max_det = 1
        result = self.vial_model(frame)
        result = result[0].boxes.data.cpu().numpy()
        if len(result) == 0:
            return None
        else:
            # vial_box = result.pred[0].cpu().numpy()[0, :4]  # if self.vial_model else result[0].cpu().numpy()
            self.vial_location = [x.astype(np.int16) for x in result[0, : 4]]
            self.vial_size = [
                int(self.vial_location[2] - self.vial_location[0]),
                int((self.vial_location[3] - self.vial_location[1]) * (1 - self.CAP_RATIO))
            ]
        return result

    @staticmethod
    def find_liquid(pred_classes, liquid_classes, all_classes):
        """
        find the first id of either 'Homo' or 'Hetero' from the class names
        :param pred_classes: [1, 2, 3]
        :param liquid_classes: ['Homo', 'Hetero']
        :param all_classes: {0: solid, 1: No solid, ....}
        :return: [index]
        """
        liquid_classes_id = [key for key, value in all_classes.items() if value in liquid_classes]
        return [index for index, c in enumerate(pred_classes) if c in liquid_classes_id]

    # NMS function for post procesing suppresion

    def custom_nms(self, bboxes):
        """
        Apply custom NMS based on class overlap rules.

        :param bboxes: Detected bounding boxes (numpy array: [x1, y1, x2, y2, conf, class_id]).
        :return: Filtered bounding boxes.
        """
        keep_indices = []
        bboxes = torch.tensor(bboxes)
        classes = [self.contents_model.names[int(idx)] for idx in bboxes[:, 5]]

        confidences = bboxes[:, 4]

        for i, bbox in enumerate(bboxes):

            suppress = False
            for j, other_bbox in enumerate(bboxes):
                if i == j:
                    continue

                iou = box_iou(bbox[:4].unsqueeze(0), other_bbox[:4].unsqueeze(0)).item()
                iou_thresholds = self.NMS_RULES.get((classes[i], classes[j]), None)
                if iou_thresholds and iou > iou_thresholds:
                    suppress = confidences[i] < confidences[j]

                if suppress:
                    break

            if not suppress:
                keep_indices.append(i)

        return bboxes[keep_indices].numpy()

    def content_detection(self, vial_frame):
        """
        Detect content in a vial frame.
        :param vial_frame: (np.ndarray) Cropped vial frame.
        :return tuple: Bounding boxes, liquid boxes, and detected class titles.
        """
        result = self.contents_model(vial_frame, max_det=4, agnostic_nms=False, conf=0.25, iou=0.25)
        bboxes = result[0].boxes.data.cpu().numpy()

        # Apply custom NMS
        bboxes = self.custom_nms(bboxes)

        pred_classes = bboxes[:, 5]  # np.array: [1, 3 ,4]
        title = " ".join([self.contents_model.names[x] for x in pred_classes])

        index = self.find_liquid(pred_classes, self.LIQUID_CONTENT, self.contents_model.names)  # [1, 3]
        liquid_boxes = [bboxes[i][:4] for i in index]
        liquid_boxes = sorted(liquid_boxes, key=lambda x: x[1], reverse=True)
        return bboxes, liquid_boxes, title

    def process_vial_frame(self,
                           vial_frame,
                           orig_frame,
                           vial_location,
                           update_od: bool = False,
                           ):
        """
        process single vial frame, detect content, draw bounding box and calculate turbidity and color
        :param vial_frame: vial frame image
        :param update_od: update object detection, True: run YOLO for this frame, False: use previous YOLO results
        """
        if update_od:
            self.content_info = self.content_detection(vial_frame)
        bboxes, liquid_boxes, title = self.content_info

        phase_data, raw_turbidity = self.calculate_value_color(vial_frame=vial_frame, liquid_boxes=liquid_boxes)

        # this part gets ugly when there is more than 1 l_bbox but for now good enough
        if self.INCLUDE_BB:
            frame = self.draw_bounding_boxes(orig_frame, bboxes, self.contents_model.names, vial_location, phase_data=phase_data, liquid_boxes = liquid_boxes, text_right=False)
        self.frame = frame
        # self.display_frame(y_values=raw_turbidity, image=frame, title=title)

        # self.fig.canvas.draw()
        # frame_image = np.array(self.fig.canvas.renderer.buffer_rgba())
        frame_image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # print(frame_image.shape) # this is 600x800
        return frame_image, raw_turbidity, phase_data

    def display_frame(self, y_values, image, title=None):
        """
        Display the image (top-left) and its turbidity values per row (top-right)
        turbidity over time (bottom-left) and color over time (bottom-right)
        :param y_values: the turbidity value per row
        :param image: vial image frame to display
        :param title: title of the image frame
        """
        # init plot
        for ax in self.axs.flat:
            ax.clear()
        ax0, ax1, ax2, ax3 = self.axs.flat

        # top left - vial frame and bounding boxes
        ax0.imshow(np.flipud(image), origin='lower')
        if title:
            ax0.set_title(title)

        # use fill between to optimize the speed 154.9857677 -> 68.15193
        x_values = np.arange(len(y_values))
        ax1.xaxis.set_label_position('top')
        ax1.set_xlabel('Turbidity per row')
        ax1.set_ylim(0, len(y_values))
        ax1.set_xlim(0, 255)
        ax1.fill_betweenx(x_values, 0, y_values[::-1], color='green', alpha=0.5)

        realtime_tick_label = None

        # bottom left - turbidity
        ax2.set_ylabel('Turbidity')
        ax2.set_xlabel('Time / min')
        ax2.plot(self.x_time, self.average_turbidity)
        ax2.set_xticks([self.x_time[0], self.x_time[-1]], realtime_tick_label)

        # bottom right - color
        ax3.set_ylabel('Color (hue)')
        ax3.set_xlabel('Time / min')
        ax3.plot(self.x_time, self.average_colors)
        ax3.set_xticks([self.x_time[0], self.x_time[-1]], realtime_tick_label)

    def calculate_value_color(self, vial_frame, liquid_boxes):
        """
        Calculate the value and color for a given vial image and bounding boxes
        :param vial_frame: the vial image
        :param liquid_boxes: the liquid boxes (["Homo", "Hetero"])
        :return: the output dict and raw turbidity per row
        """
        raw_value = []
        height, width, _ = vial_frame.shape
        hsv_image = cv2.cvtColor(vial_frame, cv2.COLOR_BGR2HSV)
        average_color = np.mean(hsv_image[:, :, 0])
        average_value = np.mean(hsv_image[:, :, 2])
        self.average_colors.append(average_color)
        self.average_turbidity.append(average_value)
        output = dict(time=self.x_time[-1], color=average_color, turbidity=average_value)
        for i in range(height):
            # Calculate the starting and ending indices for the row
            row = hsv_image[i, :]
            average_value = np.mean(row[:, 2])
            raw_value.append(average_value)
        for index, bbox in enumerate(liquid_boxes):
            # print(bbox)
            _, liquid_top, _, liquid_bottom = bbox
            start_index = int(liquid_top)
            end_index = int(liquid_bottom)
            row = hsv_image[start_index:end_index, :]
            value = np.mean(row[:, :, 2])
            color = np.mean(row[:, :, 0])
            output[f'volume_{index + 1}'] = (liquid_bottom - liquid_top) / height * 7.39
            print(f'volume_{index + 1} is', output[f'volume_{index + 1}'])
            output[f'color_{index + 1}'] = color
            output[f'turbidity_{index + 1}'] = value
            # output.append(output_per_box)
        return output, raw_value

    def save_output(self, filename):
        """
        Save the output to _per_phase.csv and _raw.csv
        :param filename: base filename
        :return: None
        """
        self.output_dataframe = pd.DataFrame(self.output)
        self.output_dataframe = self.output_dataframe.fillna(0)
        # combined_df = pd.concat([average_data, phase_data], axis=1)
        self.output_dataframe.to_csv(f"{filename}_per_phase.csv", index=False)

        # saving raw turbidity data
        turbidity_2d = np.array(self.turbidity_2d)
        turbidity_2d.T
        np.savetxt(filename + "_raw.csv", turbidity_2d, delimiter=',', fmt='%d')

    def crop_rectangle(self, image, vial_location):
        """
        crop and resize the image
        :param image: raw image capture
        :param vial_location:
        :return: cropped and resized vial frame
        """
        vial_x1, vial_y1, vial_x2, vial_y2 = vial_location
        vial_y1 = int(self.CAP_RATIO * (vial_y2 - vial_y1)) + vial_y1
        cropped_image = image[vial_y1:vial_y2, vial_x1:vial_x2]
        # cv2.resize(cropped_image, self.vial_size)
        return cv2.resize(cropped_image, self.vial_size)

    def clear_cache(self):
        """
        clear all list when starting new process
        :return: None
        """
        self.x_time = []
        self.average_colors = []
        self.average_turbidity = []
        self.content_info = None
        self.turbidity_2d = []
        self.output = []

    def run(self, frame, save_directory=None, output_name=None, fps=5,
            res=(640, 480), live_save: bool = False):
        """
        Main function to perform vial monitoring. Captures video frames from a camera or video file,
        Workflow:
        Image analysis mode:
        1. Load image 2. Detect the vial 3. Detect content 4 Save output frame as image.
        Video analysis mode:
        1. Initialize video capture (Pi Camera, webcam, or video/image file).
        2. Initialize output video write and optionally initialize video writers for saving raw frames.
        3. Detect the vial in the first frame or as needed.
        4. Process each frame:
            - Crop the vial area.
            - Perform content detection and calculate vial properties (turbidity, phase data).
            - Save processed frames and plots to video.
        5. Optionally display processed frames in real-time.
        6. Save the output data to .CSV files.
        7. Handle cleanup and resource release on completion or interruption.
        Raises:
            KeyboardInterrupt: Stops the monitoring loop when manually interrupted.

        :param source: image/video/capture
        :param save_directory: output directory, defaults to "./heinsight_output"
        :param output_name: output name, defaults to "output"
        :param fps: FPS, defaults to 5
        :param res: (realtime capturing) resolution, defaults to (1920, 1080)
        :param live_save: whether to save csv data after every frame
        :return: output over time dictionary
        """
        # ensure proper naming
        '''
        output_name = output_name or "output"
        save_directory = save_directory or './heinsight_output'
        os.makedirs(save_directory, exist_ok=True)
        output_filename = os.path.join(save_directory, output_name)
        '''
      
        self.clear_cache()
        #frame = cv2.imread(frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print("HERE1")
        result = self.find_vial(frame=frame)
        print("HERE2")
        if result is not None:
            orig_frame = frame
            vial_frame = self.crop_rectangle(image=frame, vial_location=self.vial_location)
            print("HERE3")
            self.x_time = [0]
            frame_image, _raw_turb, phase_data = self.process_vial_frame(vial_frame=vial_frame, orig_frame = orig_frame, vial_location=self.vial_location, update_od=True)
            print("HERE4")
            print(phase_data)
            # cv2.imwrite(f"{output_filename}.png", frame_image)
        else:
            print("No vial found")
        return frame_image

    @staticmethod
    def _register_colors(model_list):
        """
        register default colors for models
        :param model_list: YOLO models list
        """
        name_color_dict = {
            "Empty": (160, 82, 45),  # Brown
            "Residue": (255, 165, 0),  # Orange
            "Hetero": (255, 0, 255),  # purple
            "Homo": (255, 0, 0),  # Red
            "Solid": (0, 0, 255),  # Blue
        }
        names = [model.names.values() for model in model_list if model is not None]
        names = set(chain.from_iterable(names))
        for index, name in enumerate(names):
            if name not in name_color_dict.keys():
                name_color_dict[name] = (randint(0, 255), randint(0, 255), randint(0, 255))
        return name_color_dict

    def _set_axes(self):
        """creating plot axes"""
        ax0, ax1, ax2, ax3 = self.axs.flat
        ax0.set_position([0.21, 0.45, 0.22, 0.43])  # [left, bottom, width, height]

        ax1.set_position([0.47, 0.45, 0.45, 0.43])  # [left, bottom, width, height]
        ax2.set_position([0.12, 0.12, 0.35, 0.27])
        ax3.set_position([0.56, 0.12, 0.35, 0.27])
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    heinsight = HeinSight(vial_model_path=r"models/best_vessel.pt",
                          contents_model_path=r"models/best_content.pt", )
    #for i in range(1, 11): 
        #output = heinsight.run(rf"../examples/test_data/test_{i}.jpg", rf"../examples/volume_estimate_result", rf"result_{i}")
        #output = heinsight.run(rf"../examples/test_data/test_{i}.jpg", rf"../examples/test_data_result", rf"result_{i}")
    # output = heinsight.run(r"examples/demo.png")
    # output = heinsight.run(r"../examples/demo.png")
    # heinsight.run("C:\Users\User\Downloads\demo.png")
    # heinsight.run(r"C:\Users\User\Downloads\WIN_20240620_11_28_09_Pro.mp4")
    # heinsight.run(r"../examples/demo.png", r"../examples/test_data_result", rf"volume_test")
    # heinsight.run(r"../examples/test_data/test_video.mov")
