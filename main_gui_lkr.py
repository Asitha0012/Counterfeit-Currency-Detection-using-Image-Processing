import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image as PIL_Image, ImageTk
import cv2
import time
import os
import sys
sys.path.append('src')
import threading
import numpy as np
from evaluate_lkr import analyze_lkr_note
from align_note import align_note

class CurrencyDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FAKE CURRENCY DETECTION SYSTEM (Sri Lanka Edition)")
        self.root.geometry("1100x600")
        
        self.image_path = None
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = Frame(self.root, pady=10, padx=10)
        main_frame.pack(fill=BOTH, expand=True)

        Label(master=main_frame, text="SRI LANKA CURRENCY DETECTION SYSTEM", fg='dark blue', font="Verdana 28 bold").pack()
        Label(master=main_frame, text="This system analyzes LKR 500, 1000 and 5000 using advanced ORB + SSIM and custom Programmatic math.", fg='dark red', font="Verdana 12").pack()

        # Canvas for showing the uploaded image
        self.canvas_img = Canvas(master=main_frame, width=500, height=250, bg='white', relief=SUNKEN, borderwidth=2)
        self.canvas_img.pack(pady=20)

        # Radio buttons for denomination
        self.option = IntVar()
        self.option.set(1)
        radio_frame = Frame(main_frame)
        radio_frame.pack(pady=5)
        
        Radiobutton(master=radio_frame, text="LKR 500", variable=self.option, value=3).pack(side=LEFT, padx=10)
        Radiobutton(master=radio_frame, text="LKR 1000", variable=self.option, value=1).pack(side=LEFT, padx=10)
        Radiobutton(master=radio_frame, text="LKR 5000", variable=self.option, value=2).pack(side=LEFT, padx=10)

        # Buttons
        btn_frame = Frame(main_frame)
        btn_frame.pack(pady=10)
        
        Button(master=btn_frame, text="Select an image", fg='blue', font="Verdana 10 bold", command=self.browse_image).pack(side=LEFT, padx=10)
        Button(master=btn_frame, text="Submit", fg='green', font="Verdana 10 bold", command=self.process_image).pack(side=LEFT, padx=10)
        Button(master=btn_frame, text="Exit", fg='red', font="Verdana 10 bold", command=self.root.quit).pack(side=LEFT, padx=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if file_path:
            self.image_path = file_path
            img = cv2.imread(self.image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (500, 250))
                self.photo = ImageTk.PhotoImage(image=PIL_Image.fromarray(img))
                self.canvas_img.create_image(0, 0, image=self.photo, anchor=NW)

    def process_image(self):
        if not self.image_path: return
        
        processing_window = Toplevel(self.root)
        processing_window.title("Processing")
        processing_window.geometry("400x150")
        Label(processing_window, text="Processing! Please wait...", fg='green', font="Verdana 12 bold", pady=30).pack()
        
        val = self.option.get()
        if val == 1:
            denom_str = 'LKR_1000'
        elif val == 2:
            denom_str = 'LKR_5000'
        else:
            denom_str = 'LKR_500'
        
        def worker():
            start_time = time.perf_counter()
            flat_verdict, robust_verdict, message, score, feature_statuses, feature_images = analyze_lkr_note(self.image_path, denom_str)
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000
            
            self.root.after(0, lambda: self.on_processing_complete(processing_window, flat_verdict, robust_verdict, message, feature_statuses, feature_images, elapsed_time_ms))
            
        threading.Thread(target=worker, daemon=True).start()

    def on_processing_complete(self, processing_window, flat_verdict, robust_verdict, message, feature_statuses, feature_images, elapsed_time_ms):
        processing_window.destroy()
        self.show_results(flat_verdict, robust_verdict, message, feature_statuses, feature_images, elapsed_time_ms)

    def show_results(self, flat_verdict, robust_verdict, message, statuses, images, elapsed_time_ms):
        res_win = Toplevel(self.root)
        res_win.title("Fake Currency Detection - Result Analysis")
        res_win.geometry("1100x600")
        res_win.resizable(False, False)
        
        main_frame = Frame(res_win, relief=GROOVE, bd=1)
        main_frame.place(x=10, y=10, width=1080, height=580)
        
        canvas = Canvas(main_frame)
        master_frame = Frame(canvas)
        myscrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=myscrollbar.set)
        
        myscrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0,0), window=master_frame, anchor='nw')
        
        def scrollbar_function(event):
            canvas.configure(scrollregion=canvas.bbox("all"), width=1050, height=550)
        master_frame.bind("<Configure>", scrollbar_function)
        
        # Build the exact layout of the Euro system
        sub_frame1 = Frame(master_frame, bg='black', pady=5)
        sub_frame2 = Frame(master_frame, bg='brown', pady=5, padx=5)
        sub_frame3 = Frame(master_frame, pady=5, padx=5)
        sub_frame4 = Frame(master_frame, pady=5, padx=5)
        
        sub_frame1.grid(row=1, column=1, padx=5, pady=5)
        sub_frame2.grid(row=2, column=1, padx=5, pady=5)
        sub_frame3.grid(row=3, column=1, padx=5, pady=5)
        sub_frame4.grid(row=4, column=1, padx=5, pady=5)
        
        Label(master=sub_frame1, text="FAKE CURRENCY DETECTION SYSTEM", fg='dark blue', font="Verdana 28 bold").pack()
        
        # Input image
        c_input = Canvas(master=sub_frame2, width=675, height=300)
        c_input.pack()
        val = self.option.get()
        if val == 1: denom_str = 'LKR_1000'
        elif val == 2: denom_str = 'LKR_5000'
        else: denom_str = 'LKR_500'
        img_inp = cv2.imread(self.image_path)
        img_inp = align_note(img_inp, denom_str)
        img_inp = cv2.cvtColor(img_inp, cv2.COLOR_BGR2RGB)
        img_inp = cv2.resize(img_inp, (675, 300))
        img_inp = ImageTk.PhotoImage(PIL_Image.fromarray(img_inp))
        c_input.image = img_inp
        c_input.create_image(0, 0, anchor=NW, image=img_inp)
        
        # Grid of features
        feature_names = [
            "F1: Micro-Printing", "F2: Butterfly", "F3: Note Value", 
            "F4: Bird", "F5: Lion Emblem", "F6: Value Text", "F7: Watermark",
            "F8: Blind Dots", "F9: Asymmetric Serial", "F10: Vertical Red Serial", 
            "F11: Security Thread", "F12: Edge Lines"
        ]
        
        for i in range(4):
            for j in range(3):
                feature_num = 3*i+j
                if feature_num < 12 and feature_num < len(statuses):
                    sub_frame4.grid_rowconfigure(i, weight=1)
                    sub_frame4.grid_columnconfigure(j, weight=1)
                    
                    f_frame = Frame(master=sub_frame4, relief=RAISED, borderwidth=1, bg='light blue')
                    f_frame.grid(row=i, column=j, padx=20, pady=20, sticky="nsew")
                    
                    fr1 = Frame(f_frame, padx=3, pady=3)
                    fr2 = Frame(f_frame, bg='brown', pady=5, padx=5)
                    fr3 = Frame(f_frame)
                    fr4 = Frame(f_frame)
                    fr5 = Frame(f_frame)
                    
                    fr1.grid(row=1, column=1, padx=5, pady=5, ipadx=40)
                    fr2.grid(row=2, column=1, padx=5, pady=5)
                    fr3.grid(row=3, column=1, padx=5, pady=5)
                    fr4.grid(row=4, column=1, padx=5, pady=5)
                    fr5.grid(row=5, column=1, padx=5, pady=5)
                    
                    Label(master=fr1, text=feature_names[feature_num], fg='black', font="Verdana 12 bold").pack()
                    
                    fc = Canvas(master=fr2, width=200, height=200)
                    fc.pack()
                    
                    f_img = images[feature_num].copy()
                    h, w = f_img.shape[:2]
                    if h == 0 or w == 0:
                        f_img = np.zeros((200,200,3), dtype=np.uint8)
                        h, w = 200, 200
                        
                    aspect_ratio = w/h
                    if h > w:
                        resize_height = 200
                        resize_width = aspect_ratio * resize_height
                        img_x, img_y = (200 - resize_width)/2, 0
                    elif h < w:
                        resize_width = 200
                        resize_height = resize_width / aspect_ratio
                        img_x, img_y = 0, (200 - resize_height)/2
                    else:
                        resize_height, resize_width = 200, 200
                        img_x, img_y = 0, 0
                        
                    f_img = cv2.resize(f_img, (int(resize_width), int(resize_height)))
                    if len(f_img.shape) == 2:
                        f_img = cv2.cvtColor(f_img, cv2.COLOR_GRAY2RGB)
                    else:
                        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
                        
                    f_img = ImageTk.PhotoImage(PIL_Image.fromarray(f_img))
                    fc.image = f_img
                    fc.create_image(int(img_x), int(img_y), anchor=NW, image=f_img)
                    
                    if not hasattr(self, 'img_refs'): self.img_refs = []
                    self.img_refs.append(f_img)
                    
                    passed, msg = statuses[feature_num]
                    Label(master=fr3, text=msg, fg='dark blue', font="Verdana 11", bg='light blue', wraplength=190).pack()
                    
                    Label(master=fr4, text="", fg='dark blue', font="Verdana 11", bg='light blue').pack()
                    
                    if passed:
                        Label(master=fr5, text="Status: PASS!", fg='green', font="Verdana 11 bold", bg='light blue').pack()
                    else:
                        Label(master=fr5, text="Status: FAIL!", fg='red', font="Verdana 11 bold", bg='light blue').pack()

        passed_count = sum([1 for p, _ in statuses if p])
        current_confidence = (passed_count / 12.0) * 100.0
        header_text = f"ANALYSIS COMPLETE: {passed_count} / 12 Features Passed (Confidence Score: {current_confidence:.1f}%)"
        Label(master=sub_frame3, text=header_text, fg='dark blue', font="Verdana 16 bold", pady=5).pack()
        
        verdict_frame = Frame(master=sub_frame3)
        verdict_frame.pack(pady=5)
        
        flat_txt = "GENUINE" if flat_verdict else "COUNTERFEIT"
        flat_fg = "green" if flat_verdict else "red"
        Label(master=verdict_frame, text=f"Baseline Verdict (Flat Voting): {flat_txt}", fg=flat_fg, font="Verdana 14 bold").grid(row=1, column=1, padx=20)
        
        hybrid_txt = "GENUINE" if robust_verdict else "COUNTERFEIT"
        hybrid_fg = "green" if robust_verdict else "red"
        Label(master=verdict_frame, text=f"Robust Verdict (Two-Stage Hybrid): {hybrid_txt}", fg=hybrid_fg, font="Verdana 14 bold").grid(row=1, column=2, padx=20)
        
        Label(master=verdict_frame, text=f"Processing Speed: {elapsed_time_ms:.1f} ms", fg='black', font="Verdana 11").grid(row=2, column=1, columnspan=2, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = CurrencyDetectorApp(root)
    root.mainloop()
