import math
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, PhotoImage, ttk


class VesuviusKintsugi:
    _voxel_data_xy = None

    def __init__(self):
        self.overlay_alpha = 255
        self.data_file = None
        self.voxel_data = None
        self.photo_img = None
        self.th_layer = 0
        self.resized_img = None
        self.z_index = 0
        self.pencil_size = 20
        self.click_coordinates = None
        self.zoom_level = 2.
        self.max_zoom_level = 15
        self.drag_start_x = None
        self.drag_start_y = None
        self.canvas_position_on_dataset_x = 0
        self.canvas_position_on_dataset_y = 0
        self.pencil_cursor = None  # Reference to the circle representing the pencil size
        self.history = []  # List to store a limited history of image states
        self.max_history_size = 0  # Maximum number of states to store
        self.mask_data = None
        self.show_mask = True  # Default to showing the mask
        self.show_image = True
        self.default_masks_directory = '/src/kgl/assets/'
        # self.load_data('/src/kgl/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230702185753/20230702185753.ppm.4.h5')

        self.data_stride = 2
        self.canvas_position_on_dataset_x = 3000 // self.data_stride
        self.canvas_position_on_dataset_y = 6600 // self.data_stride
        # self.load_data('/src/kgl/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230929220924/20230929220924.ppm.surface.2.h5')
        self.load_data('/src/kgl/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230929220924/20230929220924.ppm.surface.2.copy.h5')

        # self.data_stride = 4
        # self.canvas_position_on_dataset_x = 3000 // self.data_stride
        # self.canvas_position_on_dataset_y = 6600 // self.data_stride
        # self.load_data('/src/kgl/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230929220924/20230929220924.ppm.surface.4.h5')

        self.init_ui()
        self.on_exit()


    def load_data(self, file_path=None):
        # Ask the user to select a directory containing H5FS data
        if not file_path:
            file_path = filedialog.askdirectory(title="Select H5FS File")
        if file_path:
            try:
                # Load the H5 data into the voxel_data attribute
                print("Opening H5 file:", file_path)
                self.data_file = h5py.File(file_path, 'r')
                dataset_name, dataset_shape, dataset_type, dataset_chunks = self._h5_get_first_dataset_info(self.data_file['/'])
                print("Opening dataset:", dataset_name, dataset_shape, dataset_type, dataset_chunks)
                if dataset_type != np.uint16:
                    raise Exception("Don't know how to display this dataset dtype, sorry")
                self.dataset = self.data_file.require_dataset(dataset_name, shape=dataset_shape, dtype=dataset_type, chunks=dataset_chunks)
                if self.dataset.shape[2] > 100:
                    raise Exception(f"Careful - z is {self.dataset.shape[2]}, which is too big to comfortably put in memory. Bailing out.")

                # ASSUMPTION: dataset axes are x, y, z (in that order). This is true for datasets generated with generate_surface().

                # Load debugging data to test UI without using external data sources:
                # self.dataset = (np.random.rand(5000, 2000, 20) * 0xffff).astype(np.uint16)
                # self.canvas_position_on_dataset_x = 0
                # self.canvas_position_on_dataset_y = 0

                self.mask_data = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]), dtype=np.uint8)  # axes: x, y
                print('self.mask_data.shape', self.mask_data.shape, self.mask_data.dtype)
                self.z_index = 0
                self.update_display_slice()
                self.file_name = os.path.basename(file_path)
                self.root.title(f"Vesuvius Kintsugi - {self.file_name}")
                self.bucket_layer_slider.configure(from_=0, to=self.voxel_data.shape[0] - 1)
                self.bucket_layer_slider.set(0)
                print(f"LOG: Data loaded successfully.")
            except Exception as e:
                print(f"LOG: Error loading data: {e}")

    def on_exit(self):
        if self.data_file:
            print("Closing H5 file.")
            self.data_file.close()

    # butchered from: https://stackoverflow.com/a/53340677
    def _h5_get_first_dataset_info(self, obj):
        if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
            for key in obj.keys():
                return self._h5_get_first_dataset_info(obj[key])
        elif type(obj)==h5py._hl.dataset.Dataset:
            return obj.name, obj.shape, obj.dtype, obj.chunks

    def load_mask(self):
        if self.voxel_data is None:
            print("LOG: No voxel data loaded. Load voxel data first.")
            return

        # Prompt to save changes if there are any unsaved changes
        if self.history:
            if not tk.messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Do you want to continue without saving?"):
                return

        # File dialog to select mask file
        mask_filename = filedialog.askopenfilename(
            initialdir=self.default_masks_directory,
            title="Select Masks TIFF File",
            filetypes=[("Mask TIFF files", "mask_*.tif")]
        )

        if not mask_filename:
            print("LOG: Not loading mask data, cancelled.")
            return

        try:
            im = Image.open(mask_filename)
                # self.mask_data = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))  # axes: x, y
            print('im.size', im.size)
            im = im.resize((self.dataset.shape[0], self.dataset.shape[1]), Image.NEAREST)
            self.mask_data = (np.array(im, dtype=np.uint8) / 255).T.astype(np.uint8)
            print('self.mask_data.shape', self.mask_data.shape, self.mask_data.dtype)
            print('self.dataset.shape', self.dataset.shape)

        except Exception as e:
            print(f"LOG: Error loading mask: {e}")

    def save_mask(self):
        if self.mask_data is None:
            print("LOG: No mask data to save.")
            return

        # Construct the default file name for saving
        # base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        # default_save_file_name = f"mask_{base_name}.tif"
        # Open the file dialog with the proposed file name
        save_file_path = filedialog.asksaveasfilename(
            initialdir=self.default_masks_directory,
            title="Select File to Save Mask to",
            initialfile='mask_xxx.tif',
            filetypes=[("Mask TIFF files", "mask_*.tif")]
        )

        if not save_file_path:
            print("LOG: Not saving mask data, cancelled.")
            return

        try:
            # Save the TIFF to the chosen file path
            data = self.mask_data.astype(bool).astype(np.uint8).T * 255
            print('data stats', data.min(), data.max(), data.mean())
            # Depending on the file we used, data_stride could be 1, 2, 4,... However our masks are saved always with stride 4, so let's resize as needed:
            shrink_factor = 4 // self.data_stride
            print('data.shape', data.shape)
            im = Image.fromarray(data, 'L')
            # im.thumbnail((data.shape[1] // shrink_factor, data.shape[0] // shrink_factor), Image.Resampling.LANCZOS)
            im.resize((data.shape[1] // shrink_factor, data.shape[0] // shrink_factor), Image.NEAREST)
            im.save(save_file_path, 'TIFF')
            print(f"LOG: Mask saved as TIFF in {save_file_path}")
        except Exception as e:
            print(f"LOG: Error saving mask: {e}")
            raise

    def save_state(self):
        # Save the current state of the image before modifying it
        if self.voxel_data is not None and self.max_history_size > 0:
            if len(self.history) == self.max_history_size:
                self.history.pop(0)  # Remove the oldest state
            self.history.append((self.voxel_data.copy(), self.mask_data.copy()))

    def undo_last_action(self):
        if self.history:
            self.voxel_data, self.mask_data = self.history.pop()
            self.update_display_slice()
            print("LOG: Last action undone.")
        else:
            print("LOG: No more actions to undo.")

    def on_canvas_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        # print('event', event.x, event.y)

    def on_canvas_drag(self, event):
        alt_pressed = event.state & 0x08
        if not alt_pressed:
            self.on_canvas_pencil_drag(event)
            return

        if self.drag_start_x is not None and self.drag_start_y is not None:
            # print('event', event.x, event.y)
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.canvas_position_on_dataset_x -= dx
            self.canvas_position_on_dataset_y -= dy
            self.canvas_position_on_dataset_x = min(max(self.canvas_position_on_dataset_x, 0), self.dataset.shape[0])
            self.canvas_position_on_dataset_y = min(max(self.canvas_position_on_dataset_y, 0), self.dataset.shape[1])
            self.update_display_slice()
            self.drag_start_x, self.drag_start_y = event.x, event.y

    def on_canvas_pencil_drag(self, event):
        if self.mode.get() in ["pencil", "eraser"]:
            # self.save_state()
            self.color_pixel(event.x, event.y)

    def on_canvas_release(self, event):
        self.drag_start_x = None
        self.drag_start_y = None

    def resize_with_aspect(self, image, target_width, target_height, zoom=1):
        original_width, original_height = image.size
        zoomed_width, zoomed_height = int(original_width * zoom), int(original_height * zoom)
        aspect_ratio = original_height / original_width
        new_height = int(target_width * aspect_ratio)
        new_height = min(new_height, target_height)
        return image.resize((zoomed_width, zoomed_height), Image.Resampling.NEAREST)

    def update_display_slice(self):
        if self.dataset is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # print('canvas size:', canvas_width, canvas_height)
        if canvas_width <= 1 or canvas_height <= 1:
            print("Canvas not yet initialized, can't update display slice")
            return

        needed_data_width, needed_data_height = math.ceil(canvas_width / self.zoom_level), math.ceil(canvas_height / self.zoom_level)
        x0, y0 = self.canvas_position_on_dataset_x, self.canvas_position_on_dataset_y
        # print('x0, y0:', x0, y0)
        # print('needed_data_width, needed_data_height:', needed_data_width, needed_data_height)

        # Fetch needed data from our dataset
        if self.show_image:
            # print('ccc', x0, y0)
            # Initialize voxel data as needed, but always take only the visible x/y area and the whole z, for faster scroll action:
            if self._voxel_data_xy != (x0, y0):
                self._voxel_data_xy = (x0, y0)
                self.voxel_data = self.dataset[
                    x0:x0 + needed_data_width,
                    y0:y0 + needed_data_height,
                    :,
                ] / 256  # xyz

            img_array = (self.voxel_data[:, :, self.z_index]).astype('uint16')
        else:
            img_array = np.zeros(shape=(1, needed_data_width, needed_data_height)).astype('uint16')

        img = Image.fromarray(img_array.swapaxes(0, 1)).convert('RGBA')
        # print('voxel data:', self.voxel_data.shape)
        # print('img shape:', img_array.shape)

        # Overlay the mask
        if self.mask_data is not None and self.show_mask:
            mask = self.mask_data[x0:x0+img_array.shape[0], y0:y0+img_array.shape[1]].astype(np.uint8) * self.overlay_alpha
            # print('self.mask_data:', self.mask_data.shape)
            # print('mask:', mask.shape)
            yellow = np.zeros_like(mask, dtype=np.uint8)
            yellow[:, :] = 255  # Yellow color
            mask_img = Image.fromarray(np.stack([yellow, yellow, np.zeros_like(mask), mask], axis=-1).swapaxes(0, 1), 'RGBA')

            # Overlay the mask on the original image
            img = Image.alpha_composite(img, mask_img)

        # Resize the image with aspect ratio
        img = self.resize_with_aspect(img, canvas_width, canvas_height, zoom=self.zoom_level)

        # Convert back to a format that can be displayed in Tkinter
        self.resized_img = img.convert('RGB')
        self.photo_img = ImageTk.PhotoImage(image=self.resized_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_img)
        self.canvas.tag_raise(self.z_slice_text)
        self.canvas.tag_raise(self.zoom_text)
        self.canvas.tag_raise(self.cursor_pos_text)

    def update_info_display(self):
        self.canvas.itemconfigure(self.z_slice_text, text=f"Z-Slice: {self.z_index}")
        self.canvas.itemconfigure(self.zoom_text, text=f"Zoom: {self.zoom_level:.2f}")
        if self.click_coordinates:
            try:
                _, cursor_y, cursor_x = self.calculate_image_coordinates(self.click_coordinates)
            except:
                cursor_x, cursor_y = 0, 0
            self.canvas.itemconfigure(self.cursor_pos_text, text=f"Cursor Position: ({cursor_x}, {cursor_y})")

    def on_canvas_click(self, event):
        if self.mode.get() != "pencil" and self.mode.get() != "eraser":
            raise Exception('Only pencil or eraser, sorry')

        self.save_state()
        self.color_pixel(event.x, event.y)

    def calculate_image_coordinates(self, input):
        # print('input:', input)
        if input is None:
            return 0, 0, 0  # Default values
        if isinstance(input, tuple):
            _, y, x = input
        elif hasattr(input, 'x') and hasattr(input, 'y'):
            x, y = input.x, input.y
        else:
            # Handle unexpected input types
            raise ValueError("Input must be a tuple or an event object")
        if self.voxel_data is not None:
            original_image_height, original_image_width = self.voxel_data[:, :, self.z_index].shape

            # Dimensions of the image at the current zoom level
            zoomed_width = original_image_width * self.zoom_level
            zoomed_height = original_image_height * self.zoom_level

            # Adjusting click position for panning
            pan_adjusted_x = x - self.canvas_position_on_dataset_x
            pan_adjusted_y = y - self.canvas_position_on_dataset_y

            # Calculate the position in the zoomed image
            zoomed_image_x = max(0, min(pan_adjusted_x, zoomed_width))
            zoomed_image_y = max(0, min(pan_adjusted_y, zoomed_height))

            # Scale back to original image coordinates
            img_x = int(zoomed_image_x / self.zoom_level)
            img_y = int(zoomed_image_y / self.zoom_level)

            # Debugging output
            #print(f"Clicked at: ({x}, {y}), Image Coords: ({img_x}, {img_y})")

            return self.z_index, img_y, img_x

    def color_pixel(self, canvas_x, canvas_y):
        # Left top corner is at canvas_position_on_dataset_x, canvas_position_on_dataset_y, and we have the center in canvas coords
        center_dataset_x, center_dataset_y = self.coords_canvas_to_dataset(canvas_x, canvas_y)

        if self.voxel_data is None:
            print('No voxel data')
            return

        if self.mode.get() not in ["pencil", "eraser"]:
            raise Exception('Only pencil or eraser, sorry')

        # Calculate the square bounds of the circle
        min_x = max(0, center_dataset_x - self.pencil_size)
        max_x = min(self.dataset.shape[0] - 1, center_dataset_x + self.pencil_size)
        min_y = max(0, center_dataset_y - self.pencil_size)
        max_y = min(self.dataset.shape[1] - 1, center_dataset_y + self.pencil_size)

        mask_value = 1 if self.mode.get() == "pencil" else 0
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Check if the pixel is within the circle's radius
                if math.sqrt((x - center_dataset_x) ** 2 + (y - center_dataset_y) ** 2) <= self.pencil_size:
                    self.mask_data[x, y] = mask_value
        self.update_display_slice()

    def update_pencil_size(self, val):
        self.pencil_size = int(float(val))
        self.pencil_size_var.set(f"{self.pencil_size}")

    def update_pencil_cursor(self, event):
        # Remove the old cursor representation
        if self.pencil_cursor:
            self.canvas.delete(self.pencil_cursor)
            self.update_display_slice()

        if self.mode.get() == "pencil":
            color = "yellow"
        if self.mode.get() == "eraser":
            color = "white"
        if self.mode.get() == "eraser" or self.mode.get() == "pencil":
            radius = self.pencil_size * self.zoom_level  # Adjust radius based on zoom level
            self.pencil_cursor = self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline=color, width=2)
        self.click_coordinates = (self.z_index, event.y, event.x)
        self.update_info_display()

    def scroll_or_zoom(self, event):
        # Adjust for different platforms
        ctrl_pressed = event.state & 0x04
        if sys.platform.startswith('win'):
            # Windows
            delta = event.delta
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            # Linux or macOS
            delta = 1 if event.num == 4 else -1

        if ctrl_pressed:
            self.zoom(delta)
        else:
            self.scroll(delta)

    def scroll(self, delta):
        if self.voxel_data is not None:
            # Update the z_index based on scroll direction
            delta = 1 if delta > 0 else -1
            self.z_index = max(0, min(self.z_index + delta, self.voxel_data.shape[2] - 1))
            self.update_display_slice()
            self.update_info_display()

    def zoom(self, delta):
        zoom_amount = 0.1  # Adjust the zoom sensitivity as needed
        if delta > 0:
            self.zoom_level = min(self.max_zoom_level, self.zoom_level + zoom_amount)
        else:
            self.zoom_level = max(1, self.zoom_level - zoom_amount)
        self.update_display_slice()

    def toggle_mask(self):
        # Toggle the state
        self.show_mask = not self.show_mask
        # Update the variable for the Checkbutton
        self.show_mask_var.set(self.show_mask)
        # Update the display to reflect the new state
        self.update_display_slice()
        # print(f"LOG: Label {'shown' if self.show_mask else 'hidden'}.\n")

    def toggle_image(self):
        # Toggle the state
        self.show_image = not self.show_image
        # Update the variable for the Checkbutton
        self.show_image_var.set(self.show_image)
        # Update the display to reflect the new state
        self.update_display_slice()
        print(f"LOG: Image {'shown' if self.show_image else 'hidden'}.\n")

    def update_alpha(self, val):
        self.overlay_alpha = int(float(val))
        self.update_display_slice()

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Info")
        help_window.geometry("800x700")  # Adjust size as necessary
        help_window.resizable(True, True)

        # Text widget with a vertical scrollbar
        help_text_widget = tk.Text(help_window, wrap="word", width=40, height=30)  # Adjust width and height as needed
        help_text_scrollbar = tk.Scrollbar(help_window, command=help_text_widget.yview)
        help_text_widget.configure(yscrollcommand=help_text_scrollbar.set)

        # Pack the scrollbar and text widget
        help_text_scrollbar.pack(side="right", fill="y")
        help_text_widget.pack(side="left", fill="both", expand=True)


        info_text = """Vesuvius Kintsugi: A tool for labeling 3D Zarr images for the Vesuvius Challenge (scrollprize.org).

Created by Dr. Giorgio Angelotti, Vesuvius Kintsugi is designed for efficient 3D voxel image labeling. Released under the MIT license.

Modified by kglspl.
"""
        # Insert the help text into the text widget and disable editing
        help_text_widget.insert("1.0", info_text)

    @staticmethod
    def create_tooltip(widget, text):
        # Implement a simple tooltip
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        tooltip.withdraw()

        label = tk.Label(tooltip, text=text, background="#FFFFE0", relief='solid', borderwidth=1, padx=1, pady=1)
        label.pack(ipadx=1)

        def enter(event):
            x = y = 0
            x, y, cx, cy = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def init_ui(self):
        self.root = tk.Tk()
        # self.root.iconbitmap("./icons/favicon.ico")
        self.root.title("Vesuvius Kintsugi")

        # Use a ttk.Style object to configure style aspects of the application
        style = ttk.Style()
        style.configure('TButton', padding=5)  # Add padding around buttons
        style.configure('TFrame', padding=5)  # Add padding around frames

        # Create a toolbar frame at the top with some padding
        self.toolbar_frame = ttk.Frame(self.root, padding="5 5 5 5")
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        # Create a drawing tools frame
        drawing_tools_frame = tk.Frame(self.toolbar_frame)
        drawing_tools_frame.pack(side=tk.LEFT, padx=5)

        # Load and set icons for buttons (icons need to be added)
        load_icon = PhotoImage(file='./icons/open-64.png')  # Replace with actual path to icon
        save_icon = PhotoImage(file='./icons/save-64.png')  # Replace with actual path to icon
        undo_icon = PhotoImage(file='./icons/undo-64.png')  # Replace with actual path to icon
        brush_icon = PhotoImage(file='./icons/brush-64.png')  # Replace with actual path to icon
        eraser_icon = PhotoImage(file='./icons/eraser-64.png')  # Replace with actual path to icon
        bucket_icon = PhotoImage(file='./icons/bucket-64.png')
        stop_icon = PhotoImage(file='./icons/stop-60.png')
        help_icon = PhotoImage(file='./icons/help-48.png')
        load_mask_icon = PhotoImage(file='./icons/ink-64.png')  # Replace with the actual path to icon

        self.mode = tk.StringVar(value="pencil")

        # Add buttons with icons and tooltips to the toolbar frame
        load_button = ttk.Button(self.toolbar_frame, image=load_icon, command=self.load_data)
        load_button.image = load_icon
        load_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_button, "Open Zarr 3D Image")

        load_mask_button = ttk.Button(self.toolbar_frame, image=load_mask_icon, command=self.load_mask)
        load_mask_button.image = load_mask_icon
        load_mask_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_mask_button, "Load Ink Label")

        save_button = ttk.Button(self.toolbar_frame, image=save_icon, command=self.save_mask)
        save_button.image = save_icon
        save_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(save_button, "Save Zarr 3D Label")

        undo_button = ttk.Button(self.toolbar_frame, image=undo_icon, command=self.undo_last_action)
        undo_button.image = undo_icon
        undo_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(undo_button, "Undo Last Action")

        # Brush tool button
        brush_button = ttk.Radiobutton(self.toolbar_frame, image=brush_icon, variable=self.mode, value="pencil")
        brush_button.image = brush_icon
        brush_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(brush_button, "Brush Tool")

        # Eraser tool button
        eraser_button = ttk.Radiobutton(self.toolbar_frame, image=eraser_icon, variable=self.mode, value="eraser")
        eraser_button.image = eraser_icon
        eraser_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(eraser_button, "Eraser Tool")

        self.pencil_size_var = tk.StringVar(value=str(self.pencil_size))  # Default pencil size
        pencil_size_label = ttk.Label(self.toolbar_frame, text="Pencil Size:")
        pencil_size_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        pencil_size_slider = ttk.Scale(self.toolbar_frame, from_=0, to=100, value=self.pencil_size, orient=tk.HORIZONTAL, command=self.update_pencil_size)
        pencil_size_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(pencil_size_slider, "Adjust Pencil Size")

        pencil_size_value_label = ttk.Label(self.toolbar_frame, textvariable=self.pencil_size_var)
        pencil_size_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Help button
        help_button = ttk.Button(self.toolbar_frame, image=help_icon, command=self.show_help)
        help_button.image = help_icon
        help_button.pack(side=tk.RIGHT, padx=2)
        self.create_tooltip(help_button, "Info")

        # The canvas itself remains in the center
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.z_slice_text = self.canvas.create_text(10, 10, anchor=tk.NW, text=f"Z-Slice: {self.z_index}", fill="red")
        self.zoom_text = self.canvas.create_text(10, 30, anchor=tk.NW, text=f"Zoom: {self.zoom_level:.2f}", fill="red")
        self.cursor_pos_text = self.canvas.create_text(10, 50, anchor=tk.NW, text="Cursor Position: (0, 0)", fill="red")


        # Bind event handlers
        self.canvas.bind("<Motion>", self.update_pencil_cursor)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self.on_canvas_press)
        self.canvas.bind("<B3-Motion>", self.on_canvas_pencil_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_click)  # Assuming on_canvas_click is implemented
        self.canvas.bind("<MouseWheel>", self.scroll_or_zoom)  # Assuming scroll_or_zoom is implemented
        # On Linux, Button-4 is scroll up and Button-5 is scroll down
        self.canvas.bind("<Button-4>", self.scroll_or_zoom)
        self.canvas.bind("<Button-5>", self.scroll_or_zoom)

        # Variables for toggling states
        self.show_mask_var = tk.BooleanVar(value=self.show_mask)
        self.show_image_var = tk.BooleanVar(value=self.show_image)

        # Create a frame to hold the toggle buttons
        toggle_frame = tk.Frame(self.root)
        toggle_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

        # Create toggle buttons for mask and image visibility
        toggle_mask_button = ttk.Checkbutton(toggle_frame, text="Toggle Mask", command=self.toggle_mask, variable=self.show_mask_var)
        toggle_mask_button.pack(side=tk.LEFT, padx=5, anchor='s')

        # Slider for adjusting the alpha (opacity)
        self.alpha_var = tk.IntVar(value=self.overlay_alpha)
        alpha_label = ttk.Label(toggle_frame, text="Opacity:")
        alpha_label.pack(side=tk.LEFT, padx=5, anchor='s')
        alpha_slider = ttk.Scale(toggle_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_alpha)
        alpha_slider.set(self.overlay_alpha)  # Set the default position of the slider
        alpha_slider.pack(side=tk.LEFT, padx=5, anchor='s')
        self.create_tooltip(alpha_slider, "Adjust Overlay Opacity")

        toggle_image_button = ttk.Checkbutton(toggle_frame, text="Toggle Image", command=self.toggle_image, variable=self.show_image_var)
        toggle_image_button.pack(side=tk.LEFT, padx=5, anchor='s')

        # Create a frame specifically for the sliders
        slider_frame = ttk.Frame(toggle_frame)
        slider_frame.pack(side=tk.RIGHT, padx=5)

        # Bucket Layer Slider
        self.bucket_layer_var = tk.StringVar(value="0")
        bucket_layer_label = ttk.Label(slider_frame, text="Bucket Layer:")
        bucket_layer_label.pack(side=tk.LEFT, padx=(10, 2))

        bucket_layer_value_label = ttk.Label(slider_frame, textvariable=self.bucket_layer_var)
        bucket_layer_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Create a frame for the log text area and scrollbar
        log_frame = tk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Update display canvas size, then load data from dataset and display it
        self.root.update_idletasks()
        self.update_display_slice()

        self.root.mainloop()

    def coords_canvas_to_dataset(self, canvas_x, canvas_y):
        zoom = self.zoom_level
        dataset_x, dataset_y = round(canvas_x / zoom) + self.canvas_position_on_dataset_x, round(canvas_y / zoom) + self.canvas_position_on_dataset_y
        return dataset_x, dataset_y


if __name__ == "__main__":
    editor = VesuviusKintsugi()
