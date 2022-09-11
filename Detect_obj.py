# from keras.preprocessing.image import img_to_array #old
from tensorflow.keras.utils import img_to_array #New
from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
from itertools import count
import cv2
import tkinter
from tkinter import messagebox,Button,Frame,Label,filedialog,Menu
from tkinter import *
import PIL
from PIL import ImageTk, Image
import os
from tkinter import font

class Background(Frame):
    
    def __init__(self, master, *pargs):
        Frame.__init__(self, master, *pargs)
        
        self.image = Image.open('background.png')
        self.img_copy= self.image.copy()
        self.background_image = ImageTk.PhotoImage(self.image)
        self.background = Label(self, image=self.background_image)
        self.background.pack(fill=BOTH, expand=YES)
        self.background.bind('<Configure>', self.resize_image)
        
    def resize_image(self,event):
        new_width = event.width
        new_height = event.height
        self.image = self.img_copy.resize((new_width, new_height))
        self.background_image = ImageTk.PhotoImage(self.image)
        self.background.configure(image =  self.background_image)
        
class Detect:

    def quit_window(self):
        if messagebox.askokcancel("Quit", "You want to quit now?"):
            root.destroy()
            
    def do_nothing(self):
        print("hello")
        window=tkinter.Toplevel(root)
        window.title("Identification and Classifications of Objects in Images")
        window.geometry("350x200")
#        window.pack()
#        var2 = StringVar()
#        message=Label(window,textvariable=var2)
        window.iconbitmap(r"D:\Major\FinalCodes\snowman.ico")
        with open("D:\Major\FinalCodes\help.txt","r") as f:
            Label(window,text=f.read()).pack()
#        message.place(x=0,y=0)
        
    def browsePerformed(self):
        self.path=tkinter.filedialog.askopenfilename()
        img = cv2.imread(self.path)
        file=self.path
#        orig=img.copy()
#        image2 = PIL.Image.fromarray(orig)
#        image=PIL.ImageTk.PhotoImage(image2)-
        pathlabel=tkinter.Label(root,text=self.path,relief = RAISED,bg="white",fg="black",font=('Arial',20,'italic'),width=50)
        pathlabel.place(x=50, y=160)
#        pathlabel.pack()
        
        
        
    def clear_label(self):
        self.path=""
        self.result=""
        self.resp=""
        self.restree=""
        messagebox.showinfo("info","reset successful")
        pass
    
    def display_detect(self):
        print(self.path)
        image_path=self.path
        img_pre=cv2.imread(image_path)
        imgd=cv2.resize(img_pre,(480,640))
        cv2.imshow("img",imgd)
        cv2.waitKey(0)
        imgd=imutils.resize(imgd,width=min(400,imgd.shape[1]))
        gray = cv2.cvtColor(imgd, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray",gray)
        cv2.waitKey(0)
        #        im_power_law_transformation = cv2.pow(gray,0.6)
        gray = gray/255.0
        imge=cv2.pow(gray,0.6)
        cv2.imshow('Power Law Transformation',imge)
        cv2.waitKey(0)
        # load the pre-trained network for person
        model="trained_person_res.h5"
        print("[INFO] loading pre-trained network...")
        model = load_model(model)
        orig = cv2.imread(image_path)
        image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # make predictions on the input image
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
        label = "Not Pedestrian" if pred == 0 else "Pedestrian"
        color = (0, 0, 255) if pred == 0 else (0, 255, 0)
        orig = cv2.resize(orig, (500, 500))
        self.resp=label
        print(self.resp)
#        cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
#        cv2.imshow("Results", orig)
#        cv2.waitKey(0)
        
        # load the pre-trained network for tree
        model="trained_tree.h5"
        print("[INFO] loading pre-trained network...")
        model = load_model(model)
        orig = cv2.imread(image_path)
#        cv2.imshow("orig",orig)
#        cv2.waitKey(0)
        image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
        label = "Not Tree" if pred == 0 else "\nTree"
        color = (0, 0, 255) if pred == 0 else (0, 255, 0)
        orig = cv2.resize(orig, (250, 250))
        self.restree=label
        print(self.restree)
#        cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
#        cv2.imshow("Results", orig)
#        cv2.waitKey(0)
        
        # load the pre-trained network for tree
        model="trained_car.h5"
        print("[INFO] loading pre-trained network...")
        model = load_model(model)
        orig = cv2.imread(image_path)
#        cv2.imshow("orig",orig)
#        cv2.waitKey(0)
        image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
        label = "Not Car" if pred == 0 else "Car"
        color = (0, 0, 255) if pred == 0 else (0, 255, 0)
        orig = cv2.resize(orig, (350, 350))
        self.rescar=label
        print(self.rescar)
        self.combined=self.resp+" "+self.restree+" "+self.rescar
        cv2.putText(orig, self.combined, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
        cv2.imshow("Results", orig)
        cv2.waitKey(0)
        pass
    
    def classify(self):
        if self.resp == "Pedestrian":
            path1=self.path
            ip=cv2.imread(path1)
            cv2.imshow("pedestrian",ip)
            cv2.waitKey(0)
            p_list=path1.split("/")
            print(p_list)
            path_to_class="D:/Projects/FinalCodes/Person"
            cv2.imwrite(os.path.join(path_to_class , p_list[-1]), ip)
            cv2.waitKey(0)
            pass
        if self.restree == "Tree":
            path1=self.path
            it=cv2.imread(path1)
            cv2.imshow("Tree",it)
            cv2.waitKey(0)
            p_list=path1.split("/")
            print(p_list)
            path_to_class="D:/Projects/FinalCodes/Tree"
            cv2.imwrite(os.path.join(path_to_class , p_list[-1]), it)
            cv2.waitKey(0)
            pass
        if self.rescar == "Car":
            path1=self.path
            ic=cv2.imread(path1)
            cv2.imshow("Car",ic)
            cv2.waitKey(0)
            p_list=path1.split("/")
            print(p_list)
            path_to_class="D:/Projects/FinalCodes/Car"
            cv2.imwrite(os.path.join(path_to_class , p_list[-1]), ic)
            cv2.waitKey(0)
            pass
        
        
    
    
    
    
    def __init__(self,root):
        
        var = StringVar()
        self.path=""
        self.result=""
        self.combined=""
        self.resp=""
        self.restree=""
        self.rescar=""
        var.set("Identification and Classification of Objects in Image")
        self.title=tkinter.Label(root,textvariable=var,relief = RAISED,bg="black",fg="white",font=('Arial',41,'bold'),height="1")
        self.title.place(x=0,y=0)
        self.b_style=font.Font(family='Helvetica',size=12,weight="bold")
#        
        
        comm1=self.browsePerformed
# pink #f442dc
        self.button1 = Button(root, width=25,height=2 ,text='BROWSE',command=comm1,fg="white",bg="#003366",cursor="hand1")
        self.button1['font']=self.b_style
        self.button1.place(x=950, y=150)
        
        self.button2 = Button(root, width=25,height=2, text='DETECT',command=self.display_detect,fg="white",bg="#003366",cursor="hand1")
        self.button2['font']=self.b_style
        self.button2.place(x=950, y=250)
#        
#        self.button2 = Button(root, width=14, text='DETECT CAR',command=self.display_detect_car,bd=3)
#        self.button2.place(x=80, y=300)
#        
        self.button3 = Button(root, width=25,height=2, text='RESET',command=self.clear_label,fg="white",bg="#003366",cursor="hand1")
        self.button3['font']=self.b_style
        self.button3.place(x=950, y=450)
        
        self.button4 = Button(root, width=25,height=2, text='QUIT',command=self.quit_window,fg="white",bg="#003366",cursor="hand1")
        self.button4['font']=self.b_style
        self.button4.place(x=950, y=550)
        
        
        self.button5 = Button(root ,width=25,height=2,text ='CLASSIFY',command=self.classify,fg="white",bg="#003366",cursor="hand1")
        self.button5['font']=self.b_style
        self.button5.place(x=950 ,y=350)
        
        self.menubar=Menu(root)
        self.helpmenu=Menu(self.menubar)
        self.helpmenu.add_command(label="Guidelines for Usage",command=self.do_nothing)
        self.menubar.add_cascade(label="Help",menu=self.helpmenu)
        self.exit_menu=Menu(self.menubar)
        self.menubar.add_cascade(label="Exit",menu=self.exit_menu)
        self.exit_menu.add_command(label="Exit",command=root.destroy)
        root.config(menu=self.menubar)

        

root=tkinter.Tk()
root.title("Identification and Classifications of Objects in Images")
w,h=root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.resizable(width=False,height=False)
root.iconbitmap('snowman.ico')
obj_bg=Background(root)
obj_bg.pack(fill=BOTH, expand=YES)
obj2=Detect(root)
root.mainloop()