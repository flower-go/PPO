from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import math as mt
from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the dictionary with png files")
parser.add_argument("--filename", type=str, help="Name of the resulting pdf file")
parser.add_argument("--title", type=str, help="Header of the pdf")
parser.add_argument("--columns", deafult=4, type=int, help="number of columns")
args = parser.parse_args([] if "__file__" not in globals() else None)

import os

#code from: https://stackoverflow.com/questions/67162412/creating-a-4-x-5-grid-of-images-within-report-lab-python

path = '.'
filename = 'maps_0604.pdf'
title = 'Test'


Y_POS_START = 250


def createPDF(path_to_images, document_name, document_title):
    th_size = 180*4/columns
    img_size = 115*4/columns
    x_space = th_size / (columns + 4)
    X_POS_START = x_space * 2
    x_increment = img_size + x_space
    rows_in_page = mt.ceil((A4[1]/mm)/(th_size/mm))
    print(rows_in_page)
    print("x increment {0}, th_size {1}, img_size {2}".format(x_increment,th_size,img_size))

    def rowGen(list_of_images):  # Creates a list of 4 image rows

        for i in range(0, len(list_of_images), columns):
            yield list_of_images[i:i + columns]

    def renderRow(path_to_images, row, y_pos):  # Renders each row to the page

        x_pos = X_POS_START  # starting x position
        thumb_size = th_size, th_size  # Thumbnail image size

        for i in row:
            image_filename = i.split(".")[0]  # Saves the filename for use in the draw string operation below
            img = Image.open(os.path.join(path_to_images, i))  # Opens image as a PIL object
            img.thumbnail(thumb_size)  # Creates thumbnail

            img = ImageReader(img)  # Passes PIL object to the Reportlab ImageReader

            # Lays out the image and filename
            pdf.drawImage(img, x_pos , y_pos * mm, width=img_size, height=img_size, preserveAspectRatio=True, anchor='c')
            pdf.drawCentredString((x_pos + img_size/2), (y_pos + (img_size + 5)/mm) * mm, image_filename)

            x_pos += x_increment  # Increments the x position ready for the next image

    images = [i for i in os.listdir(path_to_images) if
              i.endswith('.png')]  # Creates list of images filtering out non .jpgs
    row_layout = list(rowGen(images))  # Creates the layout of image rows

    pdf = canvas.Canvas(document_name, pagesize=A4, pageCompression=1)
    pdf.setTitle(document_title)

    rows_rendered = 0

    y_pos = Y_POS_START # Sets starting y pos
    pdf.setFont('Helvetica', 8)

    for row in row_layout:  # Loops through each row in the row_layout list and renders the row. For each 5 rows, makes a new page

        if rows_rendered == rows_in_page:

            pdf.showPage()
            pdf.setFont('Helvetica', 10)
            y_pos = 250
            rows_rendered = 0

        else:

            renderRow(path_to_images, row, y_pos)
            y_pos -= 40
            rows_rendered += 1

    pdf.save()
if __name__ == "__main__":
    #createPDF(path, filename, title, 5)
    createPDF(args.path, args.filename, args.title, args.columns)