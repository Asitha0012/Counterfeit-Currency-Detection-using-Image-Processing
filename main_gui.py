import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame, Label, NW, GROOVE, RAISED, IntVar, Radiobutton
from PIL import Image as PIL_Image
from PIL import ImageTk
import cv2
import time
import os
import sys

# Ensure the src directory is in the path so we can import evaluate
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evaluate import analyze_note

class CurrencyDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake Currency Detection System")
        self.root.geometry("1100x600")
        self.root.resizable(False, False)
        
        self.image_path = ""
        self.option = IntVar()
        self.option.set(-1)
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Recreate the exact gui_1.py interface
        top_frame = Frame(self.root, bg='black', width=1090, height=50, pady=3)
        frame1 = Frame(self.root, bg='cyan', width=1090, height=80, padx=3, pady=3)
        self.frame2 = Frame(self.root, bg='brown', width=1090, height=400, pady=5, padx=5)
        frame3 = Frame(self.root, width=1090, height=50, pady=3)
        frame4 = Frame(self.root, width=1090, height=50, pady=3)
        frame5 = Frame(self.root, bg='white', width=1090, height=20, pady=3)
        
        top_frame.grid(row=1, column=1, padx=5, pady=5)
        frame1.grid(row=2, column=1, padx=5, pady=5)
        self.frame2.grid(row=3, column=1, padx=5, pady=5)
        frame3.grid(row=4, column=1, padx=5, pady=5)
        frame4.grid(row=5, column=1, padx=5, pady=5)
        frame5.grid(row=6, column=1, padx=5, pady=5)
        
        title = Label(master=top_frame, text="FAKE CURRENCY DETECTION SYSTEM", fg='dark blue', font="Verdana 28 bold")
        title.pack()
        
        text1 = Label(master=frame1, text="This is a fake currency detection sytem. Select the currency type, browse your image file and get started!", fg='blue', font="Verdana 10")
        text1.pack()
        
        self.canvas = Canvas(master=self.frame2, width=675, height=300)
        self.canvas.pack()
        
        text2 = Label(master=frame3, text="Select currency type: ", fg='black', font="Verdana 12")
        text2.pack(side='left')
        
        R1 = Radiobutton(master=frame3, text="500", variable=self.option, value=1, font="Verdana 15")
        R1.pack(anchor=tk.W)
        R2 = Radiobutton(master=frame3, text="2000", variable=self.option, value=2, font="Verdana 15")
        R2.pack(anchor=tk.W)
        
        btn_sel = tk.Button(master=frame4, text="Select an image", command=self.select_image, font="Verdana 15 bold", fg='blue')
        btn_sel.pack(side='left', padx=10, pady=10)
        
        btn_sub = tk.Button(master=frame4, text="Submit", command=self.submit, font="Verdana 15 bold", fg='green')
        btn_sub.pack(side='left', padx=10, pady=10)
        
        btn_exit = tk.Button(master=frame4, text="Exit", command=self.root.destroy, font="Verdana 15 bold", fg='red')
        btn_exit.pack(side='left', padx=10, pady=10)
        
    def select_image(self):
        self.canvas.delete("all")
        self.image_path = tk.filedialog.askopenfilename()
        
        if len(self.image_path) > 0 and self.image_path[-4:].lower() in ['.jpg', 'jpeg', '.png']:
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (675, 300))
            img = PIL_Image.fromarray(img)
            self.current_image = ImageTk.PhotoImage(img)
            self.canvas.image = self.current_image
            self.canvas.create_image(0, 0, anchor=NW, image=self.current_image)
        else:
            messagebox.showinfo("Warning", "Please choose a valid image file!")
            
    def submit(self):
        if not self.image_path:
            messagebox.showinfo("Warning", "Please choose an image!")
        elif self.option.get() == -1:
            messagebox.showinfo("Warning", "Please choose the currency type!")
        else:
            # Process
            denom_str = '500' if self.option.get() == 1 else '2000'
            start_time = time.perf_counter()
            
            # Using the updated headless evaluate function which optionally returns images!
            flat_verdict, veto_verdict, passed_features, feature_statuses, result_list = analyze_note(self.image_path, denom_str, return_images=True)
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Open the exact same gui_2 analysis window
            self.open_analysis_window(result_list, passed_features, flat_verdict, veto_verdict, elapsed_time_ms)
            
    def open_analysis_window(self, result_list, passed_count, is_flat_genuine, is_veto_genuine, elapsed_time_ms):
        res_win = tk.Toplevel(self.root)
        res_win.title('Fake Currency Detection - Result Analysis')
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
        
        # Build the exact layout of gui_2
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
        img_inp = cv2.imread(self.image_path)
        img_inp = cv2.cvtColor(img_inp, cv2.COLOR_BGR2RGB)
        img_inp = cv2.resize(img_inp, (675, 300))
        img_inp = ImageTk.PhotoImage(PIL_Image.fromarray(img_inp))
        c_input.image = img_inp
        c_input.create_image(0, 0, anchor=NW, image=img_inp)
        
        # Grid of features
        for i in range(4):
            for j in range(3):
                feature_num = 3*i+j
                if feature_num < 12 and feature_num < len(result_list):
                    sub_frame4.grid_rowconfigure(i, weight=1)
                    sub_frame4.grid_columnconfigure(j, weight=1)
                    
                    f_frame = Frame(master=sub_frame4, relief=RAISED, borderwidth=1, bg='light blue')
                    f_frame.grid(row=i, column=j, padx=20, pady=20, sticky="nsew")
                    
                    fr1 = Frame(f_frame, padx=3, pady=3)
                    fr2 = Frame(f_frame, bg='brown', pady=5, padx=5)
                    fr3 = Frame(f_frame)
                    fr4 = Frame(f_frame)
                    fr5 = Frame(f_frame)
                    
                    fr1.grid(row=1, column=1, padx=5, pady=5, ipadx=100)
                    fr2.grid(row=2, column=1, padx=5, pady=5)
                    fr3.grid(row=3, column=1, padx=5, pady=5)
                    fr4.grid(row=4, column=1, padx=5, pady=5)
                    fr5.grid(row=5, column=1, padx=5, pady=5)
                    
                    Label(master=fr1, text=f"Feature {feature_num +1}", fg='black', font="Verdana 12 bold").pack()
                    
                    fc = Canvas(master=fr2, width=200, height=200)
                    fc.pack()
                    
                    f_img = result_list[feature_num][0].copy()
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
                    
                    if feature_num < 7:
                        avg_score = f"{result_list[feature_num][1]:.3f}"
                        text2 = f"Avg. SSIM Score: {avg_score}"
                    elif feature_num < 9:
                        line_count = f"{result_list[feature_num][1]:.3f}"
                        text2 = f"Avg. Number of lines: {line_count}"
                    elif feature_num == 9:
                        text2 = "9 characters detected!" if result_list[feature_num][1] else "Less than 9 characters detected!"
                    elif feature_num == 10:
                        text2 = "Base color & thread shift verified!" if result_list[feature_num][1] else "Invalid color/thread profiles!"
                    else:
                        text2 = "Watermark texture verified!" if result_list[feature_num][1] else "Invalid or faint watermark!"
                        
                    Label(master=fr3, text=text2, fg='dark blue', font="Verdana 11", bg='light blue').pack()
                    
                    if feature_num < 7:
                        max_score = f"{result_list[feature_num][2]:.3f}"
                        text3 = f"Max. SSIM Score: {max_score}"
                    else:
                        text3 = ""
                    Label(master=fr4, text=text3, fg='dark blue', font="Verdana 11", bg='light blue').pack()
                    
                    status_bool = result_list[feature_num][-1]
                    if status_bool:
                        Label(master=fr5, text="Status: PASS!", fg='green', font="Verdana 11 bold", bg='light blue').pack()
                    else:
                        Label(master=fr5, text="Status: FAIL!", fg='red', font="Verdana 11 bold", bg='light blue').pack()

        current_confidence = (passed_count / 12.0) * 100.0
        header_text = f"ANALYSIS COMPLETE: {passed_count} / 12 Features Passed (Confidence Score: {current_confidence:.1f}%)"
        Label(master=sub_frame3, text=header_text, fg='dark blue', font="Verdana 16 bold", pady=5).pack()
        
        verdict_frame = Frame(master=sub_frame3)
        verdict_frame.pack(pady=5)
        
        flat_txt = "GENUINE" if is_flat_genuine else "COUNTERFEIT"
        flat_fg = "green" if is_flat_genuine else "red"
        Label(master=verdict_frame, text=f"Baseline Verdict (Flat Voting): {flat_txt}", fg=flat_fg, font="Verdana 14 bold").grid(row=1, column=1, padx=20)
        
        veto_txt = "GENUINE" if is_veto_genuine else "COUNTERFEIT"
        veto_fg = "green" if is_veto_genuine else "red"
        Label(master=verdict_frame, text=f"Robust Verdict (Veto-Based): {veto_txt}", fg=veto_fg, font="Verdana 14 bold").grid(row=1, column=2, padx=20)
        
        Label(master=sub_frame3, text="Note: Veto-Based classification requires Feature 11 (Security Thread/Color) to pass regardless of other features.", fg='gray', font="Verdana 10 italic", pady=5).pack()
        Label(master=sub_frame3, text=f"Processing Speed: {elapsed_time_ms:.1f} ms", fg='black', font="Verdana 11 bold").pack()


if __name__ == "__main__":
    import numpy as np
    root = tk.Tk()
    app = CurrencyDetectorApp(root)
    root.mainloop()
