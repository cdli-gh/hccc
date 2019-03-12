from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter.messagebox import showinfo
import io
from PIL import Image, ImageTk
import keras
import imageio
import numpy as np
from scipy.misc import imresize

characters = ['UN',
              'DIM2',
              'GA',
              'DI',
              'SAL',
              'NU',
              'DU',
              'NI',
              'MA',
              'GI',
              'DA',
              'KUR',
              'MU',
              'DIB',
              'NA',
              'AB',
              'LUGAL',
              'TUR',
              'IN',
              'LAL',
              'RU',
              'RI',
              'A',
              'RA',
              'ZI',
              'UD',
              'KID-E2',
              'ZU',
              'IGI',
              'ZA',
              'GAL',
              'SI',
              'AN',
              'ME',
              'BA',
              'KI',
              'NE',
              'BI',
              'TUG2',
              'KA',
              'KU3',
              'IM',
              'LU',
              'NAM',
              'EN',
              'IB',
              'LA',
              'TA',
              'E',
              'LU2']


class Paint(object):
    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.identify_button = Button(self.root, text='identify', command=self.identify)
        self.identify_button.grid(row=0, column=4)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=5)

        self.c = Canvas(self.root, bg='white', width=640, height=640)
        self.c.grid(row=1, columnspan=6)

        self.setup()
        self.load_model()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def popup_bonus(self, cuneichar):
        load = Image.open("data/Soln/" + cuneichar +".png")
        render = ImageTk.PhotoImage(load)

        win = Toplevel()
        win.wm_title("Cuneiform Character")

        l = Label(win, image=render)
        l.image = render
        l.grid(row=0, column=0)

        b = Button(win, text=cuneichar, command=win.destroy)
        b.grid(row=1, column=0)

        b = Button(win, text="Not correct", command=win.destroy)
        b.grid(row=1, column=1)

    def popup_showinfo(self, cuneichar):
        showinfo("Cuneiform Character", cuneichar)

    def identify(self):
        white = (255, 255, 255)
        green = (0, 128, 0)
        self.c.update()
        img = self.save()
        testX = []
        testX.append(img)
        testX = np.asarray(testX).astype('float64')
        test = np.asarray([self.crop_and_downsample(x, downsample_size=32) for x in testX])
        prediction = self.predict(test)
        print prediction
        self.popup_bonus(characters[prediction])

    def predict(self, test):
        return np.argmax(self.loaded_model.predict(test))

    def save(self):
        ps = self.c.postscript(colormode='color', height=640, width=640)
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('test.png')
        return imageio.imread('test.png')

    def load_model(self):
        # load json and create model
        self.json_file = open('data/model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = keras.models.model_from_json(self.loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("data/model.h5")
        print("Loaded model from disk")

    @staticmethod
    def crop_and_downsample(originalX, downsample_size=32):
        """
        Starts with a 250 x 250 image.
        Crops to 128 x 128 around the center.
        Downsamples the image to (downsample_size) x (downsample_size).
        Returns an image with dimensions (channel, width, height).
        """
        current_dim = 640
        target_dim = 640
        margin = int((current_dim - target_dim) / 2)
        left_margin = margin
        right_margin = current_dim - margin

        # newim is shape (3, 640, 640)
        newim = originalX[:, left_margin:right_margin, left_margin:right_margin]

        # resized are shape (feature_width, feature_height, 3)
        feature_width = feature_height = downsample_size
        resized1 = imresize(newim[0:3, :, :], (feature_width, feature_height), interp="bicubic", mode="RGB")
        # resized2 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")

        # re-packge into a new X entry
        # newX = np.concatenate([resized1,resized2], axis=2)
        newX = resized1

        # the next line is EXTREMELY important.
        # if you don't normalize your data, all predictions will be 0 forever.
        newX = newX / 255.0

        return newX

    @staticmethod
    def extract_features(z):
        features = np.array([z[:, :, 0], z[:, :, 1], z[:, :, 2]])
        return features


if __name__ == '__main__':
    Paint()
