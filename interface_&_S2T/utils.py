from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PIL import Image
from openai import OpenAI


def process_image(img, resize_dims):
    return img.convert('RGBA').resize(resize_dims)


def process_frames(image_path, resize_dims, repeat=1, n_frames=None):
    im = Image.open(image_path)
    frames = []
    for frame_cnt in range(im.n_frames if n_frames is None else n_frames):
        im.seek(frame_cnt)
        frames.extend([process_image(im, resize_dims)] * repeat)
    return frames


def check_sim(i, file_map):
    return next(((1, item) for item, words in file_map.items() if i in words), (-1, ""))


class TtoS(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TtoS, self).__init__(parent)
        # Assuming parent has op_dest, alpha_dest, and file_map attributes
        self.op_dest = parent.op_dest
        self.alpha_dest = parent.alpha_dest
        self.file_map = parent.file_map
        self.scale = 2.2
        self.img_size = (int(380*self.scale), int(260*self.scale))

        # Assuming ChatGPT class is defined elsewhere
        self.chatgpt = ChatGPT(api_key="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")

        self.initUI()

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
        self.camera = QtWidgets.QLabel("Starting ...")
        self.camera.setAlignment(QtCore.Qt.AlignCenter)
        camera_layout.addWidget(self.camera_label)
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
        # Setting the button size
        self.stopCamButton.setFixedSize(200, 80)  # Width: 100, Height: 40
        self.stopCamButton.clicked.connect(self.stop_camera)
        middle_layout.addWidget(self.stopCamButton)
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
        if hasattr(self, 'cap'):
            self.cap.release()
            self.setInitialCamera()

    def start_camera(self):
        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)  # 0 for the default camera

        # Update the frame from the webcam every 30 milliseconds
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to Qt format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            return

        frame = self.gif_frames[self.cnt]
        qt_image = QtGui.QImage(frame.convert("RGBA").tobytes("raw", "RGBA"),
                                frame.size[0], frame.size[1], QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.gif_box.setPixmap(pixmap.scaled(self.img_size[0], self.img_size[1], QtCore.Qt.KeepAspectRatio))

        self.cnt += 1
        QtCore.QTimer.singleShot(15, self.gif_stream)  # Continue streaming

    def take_input(self):
        text_input = self.inputtxt.toPlainText()
        text_output = self.chatgpt(text_input)
        # Update ChatGPT response label
        self.user_question_label.setText(" - User : " + text_input)
        self.chatgpt_response_label.setText(" - ChatGPT : " + text_output)
        self.gif_frames = self.get_frames(text_output)
        self.cnt = 0
        self.gif_stream()

    def get_frames(self, text):
        all_frames = []
        for word in text.split():
            flag, sim = check_sim(word.lower(), self.file_map)
            if flag == -1:
                for letter in word:
                    if letter.isalpha():  # Check if the character is a letter
                        letter_file_path = self.alpha_dest + letter.lower() + "_small.gif"
                        all_frames += process_frames(letter_file_path, (self.img_size[0], self.img_size[1]), repeat=25)
            else:
                all_frames += process_frames(self.op_dest + sim, (self.img_size[0], self.img_size[1]))

            if word != text.split()[-1]:
                letter_file_path = self.alpha_dest + "space.gif"
                all_frames += process_frames(letter_file_path, (self.img_size[0], self.img_size[1]), repeat=50)
        return all_frames


class ChatGPT:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation = [{"role": "system", "content": "La conversation avec ChatGPT a commencé."}]

    def __call__(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation
        )

        assistant_response = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_response})
        return assistant_response
