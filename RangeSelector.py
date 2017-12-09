import os
import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class RangeSelector(threading.Thread):

    H, W = 900, 1200
    SCALE = 0.50
    CANV_H = H * SCALE
    CANV_W = W * SCALE

    default_slider_values = [
        '0', '100', '100', '10', '200', '200',
        '170', '100', '100', '180', '200', '200',
        '1',
    ]

    class HSVPicker:

        def __init__(self, h_range):
            self.h_range = h_range
            self._root = None
            self._row = 0

        def create(self, root, row):
            h_range = self.h_range
            self._root = root

            self._create_hsv_labels(row)
            self._create_hsv_range_sliders(h_range[0], h_range[1], row)

        def _create_hsv_labels(self, row):
            self._create_label("Hue", row)
            self._create_label("Saturation", row + 1)
            self._create_label("Value", row + 2)

        def _create_label(self, label_text, row):
            tk.Label(self._root, text=label_text).grid(row=row, column=0)

        def _create_hsv_range_sliders(self, lower_hue, upper_hue, row):
            self._create_hsv_group((180, 255, 255), (lower_hue, 100, 100), row, 1)
            self._create_hsv_group((180, 255, 255), (upper_hue, 200, 200), row, 2)

        def _create_hsv_group(self, limits, values, row, column):
            for i in range(3):
                self._create_slider(limits[i], values[i], row + i, column)

        def _create_slider(self, limit, value, row, column):
            slider = tk.Scale(self._root, to=limit, length=limit,
                              orient=tk.HORIZONTAL)
            slider.set(value)
            slider.grid(row=row, column=column)

    def __init__(self, image_processor):
        self._root = root = tk.Tk()

        self._image_processor = image_processor

        self._next_btn = self._create_button('Begin', self._on_next_clicked, 0, 2)
        self._create_button('Save', self._on_save_clicked, 0, 0)
        self._create_range_selector_panel(root, row=1)

        self._create_label('Smoothing filter', row=11)
        self._create_slider(row=11, column=1)
        self._create_button('Quit', self._quit, 11, 2)

        self._create_image_canvas(row=0, column=3)

        self._bind_keys_to_sliders()

        threading.Thread.__init__(self)
        self.start()

    def run(self):
        ...

    def _create_button(self, text, on_click, row, column):
        button = tk.Button(self._root, text=text, width=20, command=on_click)
        button.grid(row=row, column=column)
        return button

    def _create_range_selector_panel(self, root, row):
        self._create_separator(row)
        self._create_label('Lower HSV bounds', row + 1, 1)
        self._create_label('Upper HSV bounds', row + 1, 2)
        self.HSVPicker((0, 10)).create(root, row+2)
        self._create_separator(row + 5)
        self.HSVPicker((170, 180)).create(root, row+6)
        self._create_separator(row + 9)

    def _create_separator(self, row):
        h_line = ttk.Separator(self._root, orient=tk.HORIZONTAL)
        h_line.grid(row=row, columnspan=3, sticky=(tk.W, tk.E))

    def _create_label(self, text, row, column=0):
        tk.Label(self._root, text=text).grid(row=row, column=column)

    def _create_slider(self, row, column):
        smooth_scaler = tk.Scale(self._root, from_=1, to=20, length=100, orient=tk.HORIZONTAL)
        smooth_scaler.grid(row=row, column=column, columnspan=1)

    def _create_image_canvas(self, row, column):
        self._canvas = canvas = tk.Canvas(self._root, state='disabled',
                                          width=self.CANV_W, height=self.CANV_H)
        canvas.grid(row=row, column=column, rowspan=self._root.grid_size()[1])
        canvas.bind('<Button-1>', self._on_canvas_click)

    def _bind_keys_to_sliders(self):
        for slider in self._get_sliders():
            slider.bind('<Button-1>', lambda event: event.widget.focus_set())
            for key in ['<ButtonRelease-1>', '<KeyRelease-Left>',
                        '<KeyRelease-Right>', '<MouseWheel>']:
                slider.bind(key, self._slider_callback)

    def _slider_callback(self, event):
        if event.type == tk.EventType.MouseWheel:
            event.widget.set(event.widget.get() + event.delta/120)

        sliders = self._get_sliders()
        if event.widget is sliders[12]:  # Smoothing slider changed
            params = {'smooth': sliders[12].get()}
        elif event.widget in sliders[:6]:  # First range group changed
            params = self._build_params(0, 0, sliders)
        else:  # Second range group changed
            params = self._build_params(6, 1, sliders)

        new_img = self._image_processor.update(params)
        self._draw_image(new_img)

    @staticmethod
    def _build_params(first, range_idx, sliders):
        return {'hsv': {range_idx: {
                    'Lo': tuple([s.get() for s in sliders[first:first + 3]]),
                    'Hi': tuple([s.get() for s in sliders[first + 3:first + 6]])
        }}}

    def _get_sliders(self, first=0, last=12):
        return [s for s in self._root.grid_slaves()[::-1]
                if type(s) == tk.Scale][first:last+1]

    def _on_next_clicked(self):
        if self._next_btn['text'] == 'Begin':
            self._next_btn.config(text="Next")

        next_image = self._image_processor.get_next_image()
        if next_image is None:
            self._next_btn.config(state='disabled')
            self._canvas.delete('all')
        else:
            self._reset_sliders()
            self._draw_image(next_image)

    def _on_save_clicked(self):
        self._image_processor.save_masks()

    def _reset_sliders(self):
        sliders = self._get_sliders()
        for slider, value in zip(sliders, self.default_slider_values):
            slider.set(value)

    def _draw_image(self, image):
        self._canvas.config(state='normal')
        image = image.resize((int(self.W * self.SCALE), int(self.H * self.SCALE)))
        self._root.img = ImageTk.PhotoImage(image)
        self._canvas.create_image(0, 0, image=self._root.img, anchor=tk.NW)

    def _quit(self):
        cv2.destroyAllWindows()
        self._root.destroy()

    def _on_canvas_click(self, event):
        new_img = self._image_processor.select_contour(event.x/self.SCALE, event.y/self.SCALE)
        self._draw_image(new_img)


class ImageProcessor:

    positive_dir = 'masks/barrel/'
    negative_dir = 'masks/reddish/'
    other_dir = 'masks/other/'

    def __init__(self, image_dir):
        self._contours = []
        self._current_file = ''
        self._current_img = None
        self._filter_params = {}
        self._image_dir = image_dir
        self._contoured_image = None
        self._positive_contours = []
        self._contour_bboxes = np.empty(0, None)
        self._filters = np.empty((0, 0, 0), None)
        self._image_generator = (img for img in os.listdir(image_dir))

        self._make_dirs()

    def get_next_image(self):
        try:
            self._reset_params()
            self._current_file = next(self._image_generator)
            self._current_img = Image.open(self._image_dir + self._current_file)
            self._initialize_filters(self._current_img.height, self._current_img.width)
            return self._get_contoured_image(self._filter_params)
        except StopIteration:
            return None

    def update(self, params):
        return self._get_contoured_image(params)

    def select_contour(self, x_coord, y_coord):

        for i, bbox in enumerate(self._contour_bboxes):
            if bbox['x_min'] <= x_coord <= bbox['x_max']\
                    and bbox['y_min'] <= y_coord <= bbox['y_max']:
                self._contoured_image = \
                    cv2.drawContours(self._contoured_image, self._contours, i,
                                     (255, 255, 255), cv2.FILLED)
                self._positive_contours.append(i)
                return Image.fromarray(self._contoured_image)

    def save_masks(self):
        contours = self._contours
        h, w = self._current_img.height, self._current_img.width

        pos_selections = [contours[i] for i in self._positive_contours]
        neg_selections = [contours[i] for i in range(len(self._contours))
                          if i not in self._positive_contours]

        self._save_class_mask(pos_selections, self.positive_dir)
        self._save_class_mask(neg_selections, self.negative_dir)
        self._save_other_mask(contours, self.other_dir)

    def _make_dirs(self):
        for dir in [self.positive_dir, self.negative_dir, self.other_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def _save_class_mask(self, contours, dir):
        h, w = self._current_img.height, self._current_img.width

        mask = cv2.drawContours(np.zeros((h, w)), contours, -1, 255, cv2.FILLED)
        cv2.imwrite(f'{dir}{self._current_file[:-4]}XXX.png', np.array(mask, dtype='uint8'))

    def _save_other_mask(self, contours, dir):
        h, w = self._current_img.height, self._current_img.width

        mask = cv2.drawContours(np.ones((h, w))*255, contours, -1, 0, cv2.FILLED)
        cv2.imwrite(f'{dir}{self._current_file[:-4]}XXX.png', np.array(mask, dtype='uint8'))

    def _reset_params(self):
        self._filter_params = {'hsv': {0: {'Lo': (0, 100, 100), 'Hi': (10, 200, 200)},
                                       1: {'Lo': (170, 100, 100), 'Hi': (180, 200, 200)}},
                               'smooth': 1}

    def _get_contoured_image(self, params):
        mask = self._smooth_mask(self._create_mask(params))
        return self._add_contours_to_image(mask)

    def _initialize_filters(self, height, width):
        self._filters = np.empty((2, height, width), dtype='uint8')
        self._filters[0] = self._create_filter(0)
        self._filters[1] = self._create_filter(1)

    def _create_mask(self, params):
        self._positive_contours = []
        if 'hsv' in params:
            for hsv_idx, hsv_range in params['hsv'].items():
                for bound, hsv_value in hsv_range.items():
                    self._filter_params['hsv'][hsv_idx][bound] = hsv_value
                self._filters[hsv_idx] = self._create_filter(hsv_idx)

        if 'smooth' in params:
            self._filter_params['smooth'] = params['smooth']

        return np.bitwise_or(self._filters[0], self._filters[1], dtype='uint8')

    def _create_filter(self, filter_idx):
        hsv = self._filter_params['hsv']
        hsv_img = cv2.cvtColor(np.array(self._current_img), cv2.COLOR_RGB2HSV)

        return np.bitwise_and(np.all(hsv[filter_idx]['Lo'] <= hsv_img[:, :], axis=2),
                              np.all(hsv[filter_idx]['Hi'] >= hsv_img[:, :], axis=2))

    def _smooth_mask(self, mask):
        k_size = self._filter_params['smooth']
        kernel = np.ones((k_size, k_size))
        return cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

    def _add_contours_to_image(self, mask):
        img = np.array(self._current_img)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self._contour_bboxes = [
            {'x_min': a.min(axis=0).ravel()[0], 'y_min': a.min(axis=0).ravel()[1],
             'x_max': a.max(axis=0).ravel()[0], 'y_max': a.max(axis=0).ravel()[1]}
            for a in np.array(contours)]
        self._contoured_image = cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

        self._contours = contours
        return Image.fromarray(self._contoured_image)
