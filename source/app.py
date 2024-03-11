import tkinter as tk
from tkinter import filedialog
import subprocess


def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png;*.jpg;"), ("Video files", "*.mp4;")],
    )
    entry_path.delete(0, tk.END)
    entry_path.insert(0, file_path)


def browse_model():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("H5 files", "*.h5;")],
    )
    model_entry_path.delete(0, tk.END)
    model_entry_path.insert(0, file_path)


def run_script():
    path = entry_path.get()
    model = model_entry_path.get()
    try:
        if path and path[-3:] in ["jpg", "png"]:
            command = f'python inference.py --model_path "{model}" --img_path "{path}"'
            subprocess.run(command, shell=True)
        if path and path[-3:] in ["mp4"]:
            command = (
                f'python inference.py --model_path "{model}" --video_path "{path}"'
            )
            subprocess.run(command, shell=True)
    except Exception as e:
        print(str(e))


def run_cam():
    model = model_entry_path.get()
    command = f'python inference.py --model_path "{model}"'
    subprocess.run(command, shell=True)


# Create the main window
window = tk.Tk()
window.title("Image Detector")

# Create widgets
label_path = tk.Label(window, text="File Path:")
entry_path = tk.Entry(window, width=40)
button_browse = tk.Button(window, text="Browse", command=browse_image)
model_path = tk.Label(window, text="Model Path:")
model_entry_path = tk.Entry(window, width=40)
button_model_browse = tk.Button(window, text="Browse", command=browse_model)
button_run = tk.Button(window, text="Run Script", command=run_script)
button_run_detec = tk.Button(window, text="Run Live Cam", command=run_cam)
# Layout widgets
label_path.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
model_path.grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)

entry_path.grid(row=0, column=1, padx=10, pady=10)
model_entry_path.grid(row=1, column=1, padx=10, pady=10)

button_browse.grid(row=0, column=2, padx=10, pady=10)
button_model_browse.grid(row=1, column=2, padx=10, pady=10)

button_run.grid(row=2, column=0, columnspan=3, pady=10)
button_run_detec.grid(row=3, column=0, columnspan=3, pady=10)

# Start the main loop
window.mainloop()
