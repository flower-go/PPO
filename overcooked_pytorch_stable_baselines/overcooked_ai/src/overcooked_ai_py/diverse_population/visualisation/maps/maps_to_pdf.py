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
parser.add_argument("--columns", default=4, type=int, help="number of columns")
parser.add_argument("--prefix", default="", type=str, help="prefix ot be removed")
parser.add_argument("--postfix", default="", type=str, help="postfix to be removed")
args = parser.parse_args([] if "__file__" not in globals() else None)

import os

#code from: https://stackoverflow.com/questions/67162412/creating-a-4-x-5-grid-of-images-within-report-lab-python

path = '.'
filename = 'maps_0604.pdf'
title = 'Test'

desired_width = 100
desired_height = 100
TEXT_SIZE = 8
Y_POS_START = A4[1] - 1.1*desired_width

def createPDF(path_to_images, document_name, document_title, columns, args):
    th_size = 180*4/columns
    img_size = desired_width
    x_space = 8
    X_POS_START = x_space * 2
    x_increment = desired_width + x_space
    rows_in_page = mt.ceil((A4[1]/mm)/(th_size/mm))
    print(rows_in_page)
    print("x increment {0}, th_size {1}, img_size {2}".format(x_increment,th_size,desired_width))

    def rowGen(list_of_images):  # Creates a list of 4 image rows
        print(f"I have {len(list_of_images)} images")
        for i in range(0, len(list_of_images), columns):
            yield list_of_images[i:i + columns]

    def renderRow(path_to_images, row, y_pos):  # Renders each row to the page

        x_pos = X_POS_START  # starting x position

        for i in row:
            image_filename = i.split(".")[0]  # Saves the filename for use in the draw string operation below
            img = Image.open(os.path.join(path_to_images, i))  # Opens image as a PIL object
            width, height= img.size
            img = ImageReader(img)  # Passes PIL object to the Reportlab ImageReader
            start = int(len(args.prefix))
            print(f"image size is {width},{height}")
            end = -1*int(len(args.postfix))
            print(f"image name is {image_filename} length is {len(image_filename)} and indices are {start} and {end}")
            image_filename = image_filename.split("X")[0]
            image_filename = image_filename[start : end]
            # Lays out the image and filename
            img_ratio = width/desired_width
            pdf.drawImage(img,x_pos,y_pos, width = width/img_ratio, height = height/img_ratio)
            #pdf.drawImage(img, x_pos , y_pos * mm, preserveAspectRatio=True, anchor='c')
            print(image_filename)
            pdf.drawCentredString((x_pos + img_size/2), (y_pos - 0*img_size - TEXT_SIZE), image_filename)
            x_pos += x_increment  # Increments the x position ready for the next image
    images = [i for i in os.listdir(path_to_images) if
              i.endswith('.png')]  # Creates list of images filtering out non .jpgs
    print(f"images are {images}")
    row_layout = list(rowGen(images))  # Creates the layout of image rows

    pdf = canvas.Canvas(document_name, pagesize=A4, pageCompression=1)
    pdf.setTitle(document_title)
    pdf.drawCentredString(A4[0] / 2.0, Y_POS_START - TEXT_SIZE - 8+ desired_width*1.1,document_title) # or: pdf_text_object.textLine(text) etc.
    #print(f"Title is {document_title}")
    rows_rendered = 0

    y_pos = Y_POS_START # Sets starting y pos
    pdf.setFont('Helvetica', 8)
    print(row_layout)
    for row in row_layout:  # Loops through each row in the row_layout list and renders the row. For each 5 rows, makes a new page

        if rows_rendered == rows_in_page:

            pdf.showPage()
            pdf.setFont('Helvetica', TEXT_SIZE)
            y_pos = Y_POS_START
            rows_rendered = 0

        else:

            renderRow(path_to_images, row, y_pos)
            y_pos -= desired_width
            rows_rendered += 1
 #       break    
    pdf.save()
if __name__ == "__main__":
    #createPDF(path, filename, title, 5)
    print(f"path to images: {args.path}")
    createPDF(args.path, args.filename, args.title, args.columns,args)
