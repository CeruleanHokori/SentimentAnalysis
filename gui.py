import PySimpleGUI as sg
import os.path
from pred import predict

file_list_column = [
    [sg.Text('Say Something'), sg.InputText()],
    [sg.OK(),sg.Cancel()]
]
Variable = "Hello"
# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text(key="-TEXT-")],
    [sg.Image(filename="", key="-IMAGE-")],
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "OK":
        prediction = predict(values[0])
        if prediction == "Positive":
            window["-IMAGE-"].update(filename="sonic.png")
        else:
            window["-IMAGE-"].update(filename="shadow.png")
