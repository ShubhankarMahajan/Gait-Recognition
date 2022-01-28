from tkinter import *
from tkinter import ttk
import subprocess,os
from turtle import position
from PIL import ImageTk, Image
def cnn():
    one_shot_output = subprocess.run(['python','../Comparison/cnn.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # one_shot_output = subprocess.run(['python','sub.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    text.insert(INSERT,one_shot_output)
    text.see(END)
    text.update_idletasks()
def svm():
    one_shot_output = subprocess.run(['python','../Comparison/svm.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # one_shot_output = subprocess.run(['python','sub.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    text.insert(INSERT,one_shot_output)
    text.see(END)
    text.update_idletasks()
def rf():
    one_shot_output = subprocess.run(['python','../Comparison/rf.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # one_shot_output = subprocess.run(['python','sub.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    text.insert(INSERT,one_shot_output)
    text.see(END)
    text.update_idletasks()
def mlp():
    one_shot_output = subprocess.run(['python','../Comparison/mlp.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    # one_shot_output = subprocess.run(['python','sub.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    text.insert(INSERT,one_shot_output)
    text.see(END)
    text.update_idletasks()
def Go_Home():
    root.destroy()
    os.system("main.py")
root = Tk()
root.resizable(0,0)
root.wm_attributes('-transparentcolor','grey')
root.update_idletasks()
root.geometry('720x480')
root.title("Comparison between other Algorithms")
s = ttk.Style()
s.configure('my.TButton', font=("Roboto", 16))
bg = PhotoImage(file = "./Assests/bg.png")
label = Label( root, image = bg)
label.place(x = -2, y = 0)
text=Text(root,height=15,width=70)
scroll=Scrollbar(text)
home = ttk.Button(root,command=Go_Home)
home_img = ImageTk.PhotoImage(Image.open("Assests/Home_Button.jpg").resize((35, 35)))
home.configure(image=home_img)
home.configure(text='''Home''')
home.place(x=650,y=55)
label = ttk.Label(root, text=" Choose one of the Algorithms ",font=("Roboto", 25,"bold"))
label.pack()
label = Label(root, text="")
label.pack()
button = ttk.Button(root,text="1. Convolutional Neural Network (CNN)",width=50,command=cnn)
button.pack()
button = ttk.Button(root,text="2. Support Vector Machine (SVM)",width=50,command=svm)
button.pack()
button = ttk.Button(root,text="3. Random Forest (RF)",width=50,command=rf)
button.pack()
button = ttk.Button(root,text="4. Multilayer perceptron (MLP)",width=50,command=mlp)
button.pack()
label = ttk.Label(root, text="",font=("Roboto", 15))
label.pack()
text.configure(yscrollcommand=scroll.set)
text.config(font="Roboto")
text.pack()
root.mainloop()