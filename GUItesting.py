from tkinter import *
import tkinter.ttk as ttk
from tkinter import messagebox

pop_genres = ["Drama", "Comedy", "Thriller", "Action", "Adventure", "Romance", "Crime", "Science Fiction", "Horror",
              "Family"]


def win1():
    root = Tk()
    root.title("Movie Success Predictor")
    actors = StringVar()
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    ttk.Button(mainframe, text="Select Actors", command=selectActors(actors)).grid(column=0, row=0)
    Label(mainframe, text="Selected Actors: ").grid(row=5, column=0)
    Label(mainframe, textvariable=actors).grid(row=5, column=1)
    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)
    root.mainloop()


def selectActors(actors):
    act = Toplevel()
    a1 = StringVar()
    a2 = StringVar()
    a3 = StringVar()
    actorframe = ttk.Frame(act, padding="3 3 12 12")
    actorframe.grid(column=0, row=0, sticky=(N, W, E, S))
    ttk.Label(actorframe, text="Select Actor 1:").grid(row=1, column=1)
    a1_entry = ttk.Combobox(actorframe, value=pop_genres, textvariable=a1)
    a1_entry.grid(column=2, row=1, sticky=(W, E))
    ttk.Label(actorframe, text="Select Actor 2:").grid(row=2, column=1)
    a2_entry = ttk.Combobox(actorframe, value=pop_genres, textvariable=a2)
    a2_entry.grid(column=2, row=2, sticky=(W, E))
    ttk.Label(actorframe, text="Select Actor 3:").grid(row=3, column=1)
    a3_entry = ttk.Combobox(actorframe, value=pop_genres, textvariable=a3)
    a3_entry.grid(column=2, row=3, sticky=(W, E))

    ttk.Button(actorframe, text="Ok", command=actorframe.destroy()).grid(column=3, row=5, sticky=W)
    a1_entry.focus()


def submitActors(parent, actors, a1, a2, a3):
    if a1.get() not in pop_genres and a2.get() not in pop_genres and a3.get() not in pop_genres:
        messagebox.showinfo("Error", "Please select at least 1 valid actor")
    else:
        if a3.get() not in pop_genres:
            if a2.get() not in pop_genres:
                actors.set(a1.get())
                parent.destroy()
            else:
                if a1.get() == a2.get():
                    messagebox.showinfo("Error", "Please select up to 3 different actors")
                else:
                    actors.set(a1.get() + ", " + a2.get())
                    parent.destroy()
        else:
            if a1.get() == a2.get() or a1.get() == a3.get() or a2.get() == a3.get():
                messagebox.showinfo("Error", "Please select up to 3 different actors")
            else:
                actors.set(a1.get() + ", " + a2.get() + ", " + a3.get())
                parent.destroy()


win1()
'''



def win1():
    # this is the main/root window
    root = Tk()
    root.title("Window 1")
    startButton = Button(root, text="Start", command=win2)
    startButton.grid(row=9, column=7)
    leaveButton = Button(root, text="Quit", command=root.destroy)
    leaveButton.grid(row=1, column=1, sticky='nw')
    b1Var = StringVar()
    b2Var = StringVar()

    b1Var.set('b1')
    b2Var.set('b2')
    box1Label = Label(root, textvariable=b1Var, width=12)
    box1Label.grid(row=3, column=2)
    box2Label = Label(root, textvariable=b2Var, width=12)
    box2Label.grid(row=3, column=3)
    root.mainloop()


def win2():
    # this is the child window
    board = Toplevel()
    board.title("Window 2")
    s1Var = StringVar()
    s2Var = StringVar()
    s1Var.set("s1")
    s2Var.set("s2")
    square1Label = Label(board, textvariable=s1Var)
    square1Label.grid(row=0, column=7)
    square2Label = Label(board, textvariable=s2Var)
    square2Label.grid(row=0, column=6)


win1()
'''