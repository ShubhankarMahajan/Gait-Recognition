#Check send_mail() function for Filename when merging the project
from tkinter import*
import csv,os
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
panel = Label(root)
# panel.configure(text='''Label''')
panel.place(x=0,y=0)
label = Label(root, text="\nGait Recognition using One Shot Learning",font=("Roboto", 25,"bold"))
label.pack()
label = Label(root, text="Please choose one of the following\n\n",font=("Roboto", 15))
label.pack()
# Button(root, text="1. Validate the data",font=("Roboto", 12),command=validate).place(x=290,y=210)
button = Button(root, text="1. One Shot Learning",font=("Roboto", 12, "bold"),command=one_shot)
button.pack()
label = Label(root, text="",font=("Roboto", 15))
label.pack()
button = Button(root, text="2. Comparision between One Shot and other methods",font=("Roboto", 12, "bold"),command=compare)
button.pack()
label = Label(root, text="",font=("Roboto", 15))
label.pack()
button = Button(root, text="3. Exit",font=("Roboto", 12, "bold"),command=exit)
button.pack()
root.mainloop()