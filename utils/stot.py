#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from models.Classification_Model.suggest import Suggest
from models import KeyPointClassifier


class StoT:
    def __init__(self, parent, width=960, height=540, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5) -> None:
        self.num_classes_alphabet = 28
        self.num_classes_num = 13
        self.maj_mode = False

        self.parent = parent

        # Parameters with default values
        self.width = width
        self.height = height

        # Default value for device, as it was not specified to be a parameter
        self.device = 0
        self.use_brect = True
        self.consecutive_results = 0
        self.consecutive_non_detection = 0
        self.infer = True

        self.init_cam()

        # Init and load Model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier_alphabet = KeyPointClassifier(self.num_classes_alphabet, mode='alphabet')
        self.keypoint_classifier_num = KeyPointClassifier(self.num_classes_num, mode='number')
        self.word = ""
        self.sentence = ""
        self.suggestions = ""

        self.read_trie_dico('models/Classification_Model/dataset/Dictionary.txt')

        self.keypoint_classifier_labels_alphabet = self.read_labels(
            'models/keypoint_classifier/keypoint_classifier_label_alphabet.csv')
        self.keypoint_classifier_labels_num = self.read_labels(
            'models/keypoint_classifier/keypoint_classifier_label_num.csv')

        # Coordinate history
        self.point_history = deque(maxlen=16)

        self.mode = 0
        self.memory = self.keypoint_classifier_labels_alphabet[0]

    def stream(self):

        if not self.cap.isOpened():
            return

        self.key = cv.waitKey(10)

        self.select_mode()

        # Camera capture
        _, image = self.cap.read()

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        if self.infer:
            # Detection implementation
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                self.consecutive_non_detection = 0
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, self.point_history)
                    # Write to the dataset file
                    logging_csv(self.number, self.mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # Hand sign classification
                    if self.mode == 4:
                        hand_sign_id_num = self.keypoint_classifier_num(pre_processed_landmark_list)
                        labels = self.keypoint_classifier_labels_num
                        hand_sign_id = hand_sign_id_num

                    else:
                        hand_sign_id_alphabet = self.keypoint_classifier_alphabet(pre_processed_landmark_list)
                        labels = self.keypoint_classifier_labels_alphabet
                        hand_sign_id = hand_sign_id_alphabet

                    # Drawing part
                    debug_image = draw_bounding_rect(self.use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness, labels[hand_sign_id])
                    # recover the labels to reconstruct the words
                    if self.memory == hand_sign_id:
                        self.consecutive_results += 1
                    else:
                        self.consecutive_results = 0

                    if self.consecutive_results > 16:
                        if self.mode == 4 and hand_sign_id == 12 or self.mode != 4 and hand_sign_id == 0:
                            self.sentence += self.word + " "
                            self.word = ""
                            self.suggestions = ""
                        elif self.mode == 4 and hand_sign_id == 11 or self.mode != 4 and hand_sign_id == 27:
                            if self.word != "":
                                self.word = self.word[:-1]
                                self.suggest_word()
                            else:
                                self.sentence = self.sentence[:-1]
                                words = self.sentence.split(" ")
                                self.word = words[-1]
                                self.suggest_word()

                        else:
                            if self.mode == 4:
                                self.word += self.keypoint_classifier_labels_num[hand_sign_id]
                            else:
                                self.word += self.keypoint_classifier_labels_alphabet[hand_sign_id]
                            self.suggest_word()

                        self.consecutive_results = 0

                self.memory = hand_sign_id

            else:
                self.point_history.append([0, 0])
                self.consecutive_non_detection += 1

            debug_image = draw_point_history(debug_image, self.point_history)
            words = self.sentence + self.word
            debug_image = draw_info(debug_image, words, self.mode, self.number)

        # Screen reflection
        self.parent.frame = debug_image
        self.parent.update_frame()

        if self.consecutive_non_detection > 100 and self.infer:
            if words != "":
                self.parent.take_input(words, from_video=True)
                self.reset_words()
                self.reset_buttons_text()
                self.infer = False
                self.memory = self.keypoint_classifier_labels_alphabet[0]
            else:
                self.consecutive_non_detection = 0

    def read_labels(self, path):
        with open(path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels_alphabet = csv.reader(f)

            keypoint_classifier_labels_alphabet = [row[0] for row in keypoint_classifier_labels_alphabet]
        return keypoint_classifier_labels_alphabet

    def init_cam(self):
        # Init camera
        self.cap = cv.VideoCapture(0)

    def read_trie_dico(self, path):
        self.trie = Suggest()
        with open(path, 'r') as fichier:
            for ligne in fichier:
                mot = ligne.strip()  # Supprimer les espaces blancs
                self.trie.insert(mot)

    def suggest_word(self):
        self.suggestions = self.trie.search(self.word)[:4]
        # Mettre à jour les boutons avec les suggestions
        for i, suggestion in enumerate(self.suggestions):
            self.parent.buttons[i].setText(suggestion)

        # Si moins de 4 suggestions, effacer les textes des boutons restants
        self.reset_buttons_text()
        
        self.suggestions = " ".join(self.suggestions)

    def reset_buttons_text(self):
        for j in range(len(self.suggestions), 4):
            self.parent.buttons[j].setText('')

    def add_suggestion_to_text(self, text):
        self.word = ""
        self.sentence += text + " "
        self.suggestions = ""

    def reset_words(self):
        self.sentence = ""
        self.word = ""
        self.suggestions = ""

    # modifier ici pour mettre plus de signes et les enregistrer dans le csv
    def select_mode(self):
        self.number = -1
        if self.key == 98:  # 'b'
            self.mode = 0
        if self.key == 107:  # 'k'
            self.mode = 1
        if self.key == 104:  # 'h'
            self.mode = 2
        if self.key == 110:  # 'n'
            self.mode = 3
        if 65 <= self.key <= 90:  # A to Z (uppercase ASCII)
            self.number = self.key - 64
        if 48 <= self.key <= 57:  # 0 to 9 (numeric ASCII)
            self.number = self.key - 48
        if self.key == 48 and self.mode == 1:  # '0'
            self.number = 27
        if self.key == 100 and (self.mode == 3 or self.mode == 4):  # 'd' for delete
            self.number = 11
        if self.key == 32 and (self.mode == 3 or self.mode == 4):  # space
            self.number = 12
        elif self.key == 32:  # space
            self.number = 0
        if self.key == 109:  # 'm' key
            self.maj_mode = not self.maj_mode  # Toggle Maj mode state

            if self.maj_mode:
                self.mode = 4  # Maj mode activated
            else:
                self.mode = 0


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 28):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])

    if mode == 3 and (0 <= number <= 11):
        csv_path = 'model/keypoint_classifier/keypoint_num.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, words, mode, number):
    cv.putText(image, "Words:" + words, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 3, cv.LINE_AA)

    mode_string = ['Logging Key Point Aphabet', 'Logging Point History', 'Logging Key Point Number', 'Maj activated']
    if 1 <= mode <= 4:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image
