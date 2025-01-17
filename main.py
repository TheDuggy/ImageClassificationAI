"""
Copyright 2023 Georg Kollegger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time
from sys import argv
from colorama import Fore, Back
from colorama import init as init_colorama
from os import path
from numpy import array
from PIL import Image
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent, QFontDatabase
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStyleFactory, QMainWindow, QVBoxLayout, QLabel, QScrollArea, QLineEdit, QDialog, QWidget, QFileDialog, QFrame
from tensorflow import keras
from tensorflow import expand_dims
from tensorflow import nn
import numpy as np
import json

class App(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        init_colorama()
        self.resolve_arguments()
        uic.loadUi("./ui/Main.ui", self)
        self.img_load = False
        self.ai = {'model':  keras.models.load_model('./ai/img_classification_ai_final.h5')}
        with open('./ai/classes.json', 'r') as f:
            self.ai.update({'classes' : json.load(f)['classes-train-ds']})
        self.input_img = None
        self.hidden_frame.setVisible(False)
        self.load_img.clicked.connect(self.load)
        
        self.predict.clicked.connect(self.predict_img)
        with open(self.stylesheet_path, 'r') as qss:
            self.stylesheet_content = qss.read()

        self.setStyleSheet(self.stylesheet_content)
        self.log('INFO', 'App started successfully!')
        self.log('INFO', 'classes: ' + Back.LIGHTRED_EX + Fore.WHITE + str(self.ai['classes']) + Back.RESET)
        self.log('INFO', ' ' * 17 + '### MODEL SUMMARY BEGINNING ###')
        summary = []
        self.ai['model'].summary(print_fn=lambda s: summary.append(s))
        print(Fore.LIGHTBLUE_EX + '\n'.join(summary))
        self.log('INFO', ' ' * 20 + '### MODEL SUMMARY END ###')
    
    def load(self):
        try:
            input_img_path = QFileDialog.getOpenFileName(self, filter='Image (*);; PNG (*.png);; JPG (*.jpg);; JPEG (*.jpeg);;')[0]
            if input_img_path == '':
                return

            self.input_img_clean = Image.open(input_img_path)
            self.input_img_predict = keras.utils.img_to_array(keras.utils.load_img(input_img_path, target_size=(224, 224)))

            self.findChild(QLabel, 'img_type').setText(self.input_img_clean.format if self.input_img_clean.format != '' else 'None')
            self.findChild(QLabel, 'img_path').setText(input_img_path)

            self.hidden_frame.setVisible(False)
            self.img_load = True
            self.render_image()
        except Exception as e:
            self.log('ERROR', str(e))
            warning = QDialog()
            uic.loadUi('./ui/Warning.ui', warning)
            warning.setWindowTitle('Failed to open image!')
            warning.setStyleSheet(self.stylesheet_content)
            warning.ok.clicked.connect(warning.close)
            warning.exec()

    def resolve_arguments(self):
        self.stylesheet_path = './ui/style/default_light.qss'
        if len(argv) == 2:
            style = argv[1].split('=')
            if len(style) == 2:
                if style[0] == '--stylesheet':
                    if path.exists(style[1]) and path.isfile(style[1]):
                        self.stylesheet_path = style[1]
                    else:
                        self.log('WARNING', 'File ' + style[1] + ' not found! Using default ' + self.stylesheet_path) 
                else:
                    self.log('WARNING', 'Invalid argument ' + style + '! Ignoring it...')
            else:
                self.log('WARNING', 'Invalid argument ' + style + '! Ignoring it...')

        self.log('INFO', 'Stylesheet: ' + Back.LIGHTRED_EX + Fore.WHITE + path.realpath(self.stylesheet_path) + Back.RESET)

    def log(self, level: str, msg: str, reset=True):
        if level == 'INFO': 
            print(f'{Fore.LIGHTGREEN_EX}[+] ', end='')
        elif level == 'ERROR':
            print(f'{Fore.LIGHTRED_EX}[!] ', end='')
        elif level == 'WARNING':
            print(f'{Fore.LIGHTYELLOW_EX}[::] ', end='')
        else:
            return

        if reset:
            print(msg + Fore.RESET)


    def list_all_classes(self,  sorted_classes: dict):
        scroll_view_widget = self.hidden_frame.findChild(QScrollArea, 'prediction_list').findChild(QWidget, 'list')
        if scroll_view_widget.layout() is not None:
            for i in reversed(range(scroll_view_widget.layout().count())): 
                scroll_view_widget.layout().itemAt(i).widget().setParent(None)
        list = QVBoxLayout() if scroll_view_widget.layout() is None else scroll_view_widget.layout()
        self.hidden_frame.findChild(QScrollArea, 'prediction_list').verticalScrollBar().setStyle(QStyleFactory.create('Fusion'))
        del sorted_classes[max(sorted_classes, key=sorted_classes.get)]
        for class_name in sorted_classes.keys():
            el = QWidget()
            uic.loadUi('./ui/list_element.ui', el)
            el.findChild(QLabel, 'list_prediction').setText(class_name)
            el.findChild(QLabel, 'list_accuracy').setText("{:.6f}%".format(sorted_classes[class_name] * 100))
            list.addWidget(el)

        if scroll_view_widget.layout() is None:
            scroll_view_widget.setLayout(list)

    def predict_img(self):
        test_img_array = self.input_img_predict
        test_img_array = expand_dims(test_img_array, 0)
        begin_time = time()
        predictions = self.ai['model'].predict(test_img_array, verbose=0)
        time_taken = time() - begin_time
        self.log('INFO', 'Prediction {:.3f}s'.format(time_taken))
        score = nn.softmax(predictions[0])
        self.hidden_frame.findChild(QLabel, 'prediction').setText(str(self.ai['classes'][np.argmax(score, axis=-1)]).capitalize())
        self.hidden_frame.findChild(QLabel, 'accuracy').setText("{:.2f}%".format( 100 * np.max(score)))

        unsorted_classes = {}

        probabilities = np.array(score)

        for i in range(0, len(probabilities)):
            unsorted_classes.update({self.ai['classes'][i] : probabilities[i]})

        sorted_classes = {k: v for k, v in sorted(unsorted_classes.items(), key=lambda item: item[1])}

        self.list_all_classes(dict(reversed(list(sorted_classes.items()))))
        self.hidden_frame.setVisible(True)
        
    def resizeEvent(self, a0: QResizeEvent) -> None:
        t = super().resizeEvent(a0)
        if self.img_load:
            self.render_image()
        return t

    def render_image(self):
        input_img = self.input_img_clean.convert('RGB')

        max_w = self.right_frame.findChild(QLabel, 'img').geometry().width()
        max_h = self.right_frame.findChild(QLabel, 'img').geometry().height()

        curr_w = input_img.width
        curr_h = input_img.height
        ratio = curr_w / curr_h

        new_w = 0
        new_h = 0

        if curr_h > curr_w:
            new_h = max_h
            new_w = int(round(new_h * ratio))
        elif curr_w > curr_h:
            new_w = max_w
            new_h = int(round(new_w / ratio))
        else:
            if max_w > max_h:
                new_h = max_h
                new_w = max_h
            else:
                new_h = max_w
                new_w = max_w

        input_img = input_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img = QImage(array(input_img), input_img.width, input_img.height, 3 * input_img.width, QImage.Format.Format_RGB888)
        self.right_frame.findChild(QLabel, 'img').setPixmap(QPixmap.fromImage(img))

if __name__ == '__main__':
    app = QApplication([])
    dialog = App()
    QFontDatabase.addApplicationFont('./ui/style/font/JUST Sans/JUST Sans Regular.otf')
    QFontDatabase.addApplicationFont('./ui/style/font/JUST Sans/JUST Sans ExBold.otf')
    app.setStyle(QStyleFactory.create('Fusion'))
    dialog.show()
    exit(app.exec())