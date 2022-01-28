import tkinter as tk
from tkinter import ttk
def one_shot():
    print("Yeah")
root = tk.Tk()
tk.Button(root, text="Test 1",command=one_shot).pack()
ttk.Button(root, text="1. One Shot Learning",command=one_shot).pack()
root.mainloop()