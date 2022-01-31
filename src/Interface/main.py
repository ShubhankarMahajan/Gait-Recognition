#Check send_mail() function for Filename when merging the project
from tkinter import*
from tkinter import ttk
import csv,os
from turtle import width
from PIL import Image, ImageTk
from itertools import count, cycle
class ImageLabel(Label):
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)
def exit():
    root.destroy()
def compare():
    root.destroy()
    os.system("Compare.py")
def one_shot():
    root.destroy()
    os.system("One_Shot.py")
# def faq():
#     root.destroy()
#     os.system("FAQ.py")
root = Tk()
root.resizable(0,0)
root.wm_attributes('-transparentcolor','grey')
root.geometry('720x480')
root.title("Home")
s = ttk.Style()
s.configure('my.TButton', font=("Roboto", 16))
panel = Label(root)
# panel.configure(text='''Label''')
panel.place(x=0,y=0)
bg = PhotoImage(file = "./Assests/bg.png")
label = Label( root, image = bg)
label.place(x = -2, y = 0)
label = ttk.Label(root, text=" Batch A-05 ")
label.configure(font=("Roboto", 20, "bold"))
label.pack()
label = ttk.Label(root, text="\nGait Recognition using One Shot Learning")
label.configure(font=("Roboto", 20, "bold"))
label.pack()
label = ttk.Label(root, text="Please choose one of the following")
label.configure(font=("Roboto", 16, "bold"))
label.pack()
lbl = ImageLabel(root)
lbl.pack()
lbl.load('./Assests/Working.gif')
label = ttk.Label(root, text="")
label.pack()
button = ttk.Button(root, text=" One Shot Learning ",style='my.TButton',width=50,command=one_shot)
button.pack()
label = ttk.Label(root, text="")
label.pack()
button = ttk.Button(root, text=" Compare Gait Recognition with other methods ",width=50,style='my.TButton',command=compare)
button.pack()
label = ttk.Label(root, text="")
label.pack()
button = ttk.Button(root, text=" Exit ",style='my.TButton',width=50,command=exit)
button.pack()
label = ttk.Label(root, text="")
label.pack()
root.mainloop()
