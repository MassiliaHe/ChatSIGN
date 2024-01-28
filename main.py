import sys
import threading
from PyQt5 import QtWidgets, QtGui, QtCore

from utils.ttos import TtoS
from utils.iw import IW

class MainView(QtWidgets.QMainWindow):
    initializationComplete = QtCore.pyqtSignal()  # Signal for completion of initialization

    def __init__(self, *args, **kwargs):
        super(MainView, self).__init__(*args, **kwargs)
        self.op_dest = "filtered_data/"
        self.alpha_dest = "alphabet/transparent/"
        self.editFiles = [item for item in os.listdir(self.op_dest) if ".webp" in item]
        self.file_map = {i: i.replace(".webp", "").split() for i in self.editFiles}
        # self.setStyleSheet("background-image: url(images/logo2.png);")

        self.initUI()

    def initUI(self):
        self.setWindowTitle('ChatSIGN')
        self.quit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(self.close)

        self.init_window()

        self.centerOnScreen(2)

        self.inialization_window = IW()
        self.setCentralWidget(self.inialization_window)
        self.show()

        self.main_view = TtoS(self)

        # Connect the signal to the slot
        self.initializationComplete.connect(self.switch_to_main_view)

        # Run initialization in a separate thread
        init_thread = threading.Thread(target=self.initialize)
        init_thread.start()

    def initialize(self):
        # Initialize the models that take time
        self.main_view.init_hard()
        # Emit signal when initialization is complete
        self.initializationComplete.emit()

    @QtCore.pyqtSlot()
    def switch_to_main_view(self):
        # Now executes in the main thread
        self.setCentralWidget(self.main_view)

    def centerOnScreen(self, div=4):
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // div
        y = (screen.height() - size.height()) // div
        self.move(x, y)

    def launch_fullscreen(self):
        self.showFullScreen()

    def applyStyle(self):
        # Chargement des styles CSS à partir d'un fichier externe
        with open('style.css', 'r') as f:
            self.setStyleSheet(f.read())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainView()
    mainWin.centerOnScreen(4)
    mainWin.setWindowIcon(QtGui.QIcon("images\logo.ico"))
    sys.exit(app.exec_())
