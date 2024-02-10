import os
import cv2
import unicodedata
from PyQt5 import QtCore, QtGui, QtWidgets
import threading

from utils.stot import StoT
from utils.chatgpt import ChatGPT


def convert_cv2_frame_to_qt_pixmap(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QtGui.QPixmap(QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888))


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def load_all_letter_frames(alpha_dest, img_size):
    letters_frames = {}
    for filename in os.listdir(alpha_dest):
        if filename.endswith(".mp4"):
            # Extraire le nom du fichier sans l'extension .mp4
            letter = os.path.splitext(filename)[0]
            letter_path = os.path.join(alpha_dest, filename)
            letters_frames[letter] = process_frames(letter_path, img_size)
    return letters_frames

def process_frames(video_path, resize_dims):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Recadrer et redimensionner la frame en une seule étape
        # x, y, w, h = 150, 50, resize_dims[1], resize_dims[0]
        cropped_and_resized_frame = cv2.resize(frame, resize_dims)

        frames.append(cropped_and_resized_frame)

    cap.release()
    return frames


def check_sim(i, file_map):
    return next(((1, item) for item, words in file_map.items() if i in words), (-1, ""))


class TtoS(QtWidgets.QWidget):
    def __init__(self, parent=None, scale=2.0):
        super(TtoS, self).__init__(parent)
        # Assuming parent has op_dest, alpha_dest, and file_map attributes
        self.op_dest = "data/words_mp4/"
        self.alpha_dest = "data/alphabet_mp4/"
        self.editFiles = [item for item in os.listdir(self.op_dest) if ".mp4" in item]
        self.file_map = {i: i.replace(".mp4", "").split() for i in self.editFiles}
        self.scale = scale
        self.img_size = (int(320*self.scale), int(260*self.scale))
        self.frame = None
        self.mode = 'word'

        # Assuming ChatGPT class is defined elsewhere
        self.chatgpt = ChatGPT(api_key="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")

        self.initUI()

    def init_hard(self):
        self.alphabet_frames = load_all_letter_frames(self.alpha_dest, self.img_size)
        self.stot = StoT(self)
        # Initialize the video capture
        self.stot.init_cam()

    def initUI(self):
        self.cnt = 0
        self.gif_frames = []

        # Main vertical layout for the whole window
        windows_layout = QtWidgets.QVBoxLayout(self)

        # Title at the top
        titleLabel = QtWidgets.QLabel("Chat sign language")
        titleLabel.setFont(QtGui.QFont("Verdana", 12))
        titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        windows_layout.addWidget(titleLabel)

        main_layout = QtWidgets.QHBoxLayout()

        # Camera section on the left
        camera_layout = QtWidgets.QVBoxLayout()
        self.camera_label = QtWidgets.QLabel("Camera Feed Here")
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        # Création des boutons pour les suggestions
        self.camera_suggestions = QtWidgets.QLabel("Suggestions : ")
        self.buttons = [QtWidgets.QPushButton() for _ in range(4)]
        button_layout = QtWidgets.QHBoxLayout()
        for btn in self.buttons:
            btn.setText('')  # Initialiser les boutons sans texte
            button_layout.addWidget(btn)
            # Connecter le signal clicked à la méthode de slot avec le texte du bouton
            btn.clicked.connect(lambda _, b=btn: self.stot.add_suggestion_to_text(b.text()))

        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_suggestions)
        camera_layout.addLayout(button_layout)
        self.camera = QtWidgets.QLabel("Starting ...")
        self.camera.setAlignment(QtCore.Qt.AlignCenter)
        camera_layout.addWidget(self.camera)
        main_layout.addLayout(camera_layout)

        # Horizontal layout for camera and image sections
        middle_layout = QtWidgets.QVBoxLayout()
        middle_layout.addWidget(QtWidgets.QLabel("Enter Text:"))
        self.inputtxt = QtWidgets.QTextEdit()
        self.inputtxt.setFixedHeight(40)
        middle_layout.addWidget(self.inputtxt)
        sendButton = QtWidgets.QPushButton("Send")
        sendButton.clicked.connect(self.take_input)
        middle_layout.addWidget(sendButton)
        self.user_question_label = QtWidgets.QLabel(" - User : ")
        self.user_question_label.setWordWrap(True)
        middle_layout.addWidget(self.user_question_label)
        self.chatgpt_response_label = QtWidgets.QLabel(" - ChatGPT : ")
        self.chatgpt_response_label.setWordWrap(True)
        middle_layout.addWidget(self.chatgpt_response_label)
        self.startCamButton = QtWidgets.QPushButton("Start Camera")
        # Apply styles and size to startCamButton as before
        self.startCamButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;  /* Green background */
                color: white;  /* White text */
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
                    }
                QPushButton:pressed {
                    background-color: #45C35E;  /* Slightly darker green */
                    border-style: inset;
                    }
                """)
        # Setting the button size
        self.startCamButton.setFixedSize(200, 80)  # Width: 100, Height: 40
        self.startCamButton.clicked.connect(self.start_camera)
        middle_layout.addWidget(self.startCamButton)
        self.stopCamButton = QtWidgets.QPushButton("Stop Camera")
        # Apply styles and size to startCamButton as before
        self.stopCamButton.setStyleSheet("""
            QPushButton {
                background-color: #E53935;  /* Green background */
                color: white;  /* White text */
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
                    }
                QPushButton:pressed {
                    background-color: #D32F2F;  /* Slightly darker green */
                    border-style: inset;
                    }
                """)
        self.stopAnimButton = QtWidgets.QPushButton("Stop Animation")
        self.stopAnimButton.setStyleSheet("""
            QPushButton {
                background-color: #E53935;  /* Green background */
                color: white;  /* White text */
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
                    }
                QPushButton:pressed {
                    background-color: #D32F2F;  /* Slightly darker green */
                    border-style: inset;
                    }
                """)
        # Toggle button setup
        self.toggle_button = QtWidgets.QPushButton("Mode: Word by Word", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                    background-color: #1E90FF;  /* Green background */
                    color: white;  /* White text */
                    border-style: outset;
                    border-width: 2px;
                    border-radius: 10px;
                    border-color: beige;
                    font: bold 14px;
                    min-width: 10em;
                    padding: 6px;
                }
                QPushButton:checked {
                    background-color: #1A7FDE;  /* Slightly darker green */
                    color: white;
                    border-style: inset;
                    border-width: 2px;
                    border-radius: 10px;
                    border-color: beige;
                    font: bold 14px;
                    min-width: 10em;
                    padding: 6px;
                }
        """)
        # Setting the button size
        self.stopCamButton.setFixedSize(200, 80)  # Width: 100, Height: 40
        self.stopCamButton.clicked.connect(self.stop_camera)
        self.stopAnimButton.setFixedSize(200, 80)  # Width: 100, Height: 40
        self.stopAnimButton.clicked.connect(self.stop_animation)
        self.toggle_button.setFixedSize(200, 80)  # Width: 100, Height: 40
        self.toggle_button.clicked[bool].connect(self.toggle_mode)
        middle_layout.addWidget(self.stopCamButton)
        middle_layout.addWidget(self.stopAnimButton)
        middle_layout.addWidget(self.toggle_button)
        main_layout.addLayout(middle_layout)

        # Image section on the right
        image_layout = QtWidgets.QVBoxLayout()
        self.gif_box_label = QtWidgets.QLabel("Image Display Here")
        self.gif_box_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gif_box = QtWidgets.QLabel("Starting ...")
        self.gif_box.setAlignment(QtCore.Qt.AlignCenter)
        image_layout.addWidget(self.gif_box_label)
        image_layout.addWidget(self.gif_box)
        main_layout.addLayout(image_layout)
        main_layout.setStretch(0, 3)  # 1st number is the index, 2nd is the stretch factor
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 3)

        # Add the main horizontal layout to the window's layout
        windows_layout.addLayout(main_layout)

        # Set the initial logo (if implemented)
        self.setInitialLogo()
        self.setInitialCamera()

    def stop_camera(self):
        # Release the video capture when the widget is closed
        if hasattr(self.stot, 'cap'):
            self.timer.stop()
            self.stot.cap.release()
            self.setInitialCamera()
            self.stot.reset_words()

    def stop_animation(self):
        # Release the video capture when the widget is closed
        if self.cnt > 0:
            self.cnt = len(self.gif_frames)

    def toggle_mode(self, checked):
        if not checked:
            self.toggle_button.setText("Mode: Word by Word")
            # Switch to 'word by word' mode
            self.mode = "word"
        else:
            self.toggle_button.setText("Mode: Letter by Letter")
            # Switch to 'letter by letter' mode
            self.mode = "letter"

    def start_camera(self):
        if not self.stot.cap.isOpened():
            # Run initialization in a separate thread
            init_thread = threading.Thread(target=self.stot.init_cam)
            init_thread.start()
        # Update the frame from the webcam every 20 milliseconds
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.stot.stream)
        self.timer.start(20)

    def update_frame(self):
        # Convert the frame to Qt format
        if self.frame is not None:
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_frame = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_frame)
            self.camera.setPixmap(pixmap.scaled(self.img_size[0], self.img_size[1], QtCore.Qt.KeepAspectRatio))

    def setInitialLogo(self):
        self.logo1 = QtGui.QPixmap("images/logo1.png")  # Replace with the path to your logo
        self.gif_box.setPixmap(self.logo1.scaled(
            self.img_size[0], self.img_size[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def setInitialCamera(self):
        self.logo2 = QtGui.QPixmap("images/logo2.png")  # Replace with the path to your logo
        self.camera.setPixmap(self.logo2.scaled(
            self.img_size[0], self.img_size[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def gif_stream(self):
        if self.cnt >= len(self.gif_frames):
            # Show the logo at the end of the stream
            self.gif_box.setPixmap(self.logo1.scaled(
                self.img_size[0], self.img_size[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.stot.infer = True
            return

        frame = self.gif_frames[self.cnt]
        # qt_image = QtGui.QImage(frame.convert("RGBA").tobytes("raw", "RGBA"),
        #                         frame.size[0], frame.size[1], QtGui.QImage.Format_RGBA8888)
        # pixmap = QtGui.QPixmap.fromImage(qt_image)
        pixmap = convert_cv2_frame_to_qt_pixmap(frame)
        self.gif_box.setPixmap(pixmap)

        self.cnt += 1
        QtCore.QTimer.singleShot(15, self.gif_stream)  # Continue streaming

    def take_input(self, words=None, from_video=False):
        if not from_video:
            text_input = self.inputtxt.toPlainText()
        else:
            text_input = words
        self.text_output = self.chatgpt(text_input)
        # Update ChatGPT response label
        self.user_question_label.setText(" - User : " + text_input)
        self.chatgpt_response_label.setText(" - ChatGPT : " + self.text_output)
        self.text_output = remove_accents(self.text_output)
        self.gif_frames = self.get_frames()
        self.cnt = 0
        self.gif_stream()

    def get_frames(self):
        all_frames = []

        for word in self.text_output.split():
            if self.mode == 'letter':
                for letter in word:
                    if letter.isalpha():  # Check if the character is a letter
                        all_frames += self.alphabet_frames[letter.lower()]
                if word != self.text_output.split()[-1]:
                    all_frames.extend([all_frames[-1] for _ in range(50)])
            else:
                flag, sim = check_sim(word.lower(), self.file_map)
                if flag != -1:
                    all_frames += process_frames(self.op_dest + sim, (self.img_size[0], self.img_size[1]))

        return all_frames
