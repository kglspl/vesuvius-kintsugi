import tkinter as tk
from tkinter import filedialog, PhotoImage, ttk
from PIL import Image, ImageTk
import numpy as np
import h5py
from collections import deque
import threading
import math
import os
import sys

class VesuviusKintsugi:
    _voxel_data_xy = None

    def __init__(self):
        self.overlay_alpha = 255
        self.barrier_mask = None  # New mask to act as a barrier for flood fill
        self.editing_barrier = False  # False for editing label, True for editing barrier
        self.max_propagation_steps = 100  # Default maximum propagation steps
        self.show_barrier = True
        self.data_file = None
        self.voxel_data = None
        self.photo_img = None
        self.th_layer = 0
        self.resized_img = None
        self.z_index = 0
        self.pencil_size = 0
        self.click_coordinates = None
        self.threshold = [10]
        self.zoom_level = 1
        self.max_zoom_level = 15
        self.drag_start_x = None
        self.drag_start_y = None
        self.image_position_x = 0
        self.image_position_y = 0
        self.pencil_cursor = None  # Reference to the circle representing the pencil size
        self.flood_fill_active = False  # Flag to control flood fill
        self.history = []  # List to store a limited history of image states
        self.max_history_size = 3  # Maximum number of states to store
        self.mask_data = None
        self.show_mask = True  # Default to showing the mask
        self.show_image = True
        # self.load_data('/src/kgl/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230702185753/20230702185753.ppm.4.h5')

        data_stride = 4
        self.image_position_x = 3000 // data_stride
        self.image_position_y = 6600 // data_stride
        self.load_data('/src/kgl/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230929220924/20230929220924.ppm.surface.4.h5')

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
                # self.image_position_x = 0
                # self.image_position_y = 0

                self.mask_data = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))  # axes: x, y
                print('mask shape', self.mask_data.shape)
                self.barrier_mask = np.zeros_like(self.voxel_data)
                self.z_index = 0
                if self.voxel_data is not None:
                    self.threshold = [10 for _ in range(self.voxel_data.shape[0])]
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
        mask_file_path = filedialog.askdirectory(
            title="Select Label Zarr File")


        if mask_file_path:
            try:
                loaded_mask = np.array(zarr.open(mask_file_path, mode='r'))
                if loaded_mask.shape == self.voxel_data.shape:
                    self.mask_data = loaded_mask
                    self.update_display_slice()
                    print("LOG: Label loaded successfully.")
                else:
                    print("LOG: Error: Label dimensions do not match the voxel data dimensions.")
            except Exception as e:
                print(f"LOG: Error loading mask: {e}")

    def save_image(self):
        if self.mask_data is not None:
            # Construct the default file name for saving
            base_name = os.path.splitext(os.path.basename(self.file_name))[0]
            default_save_file_name = f"{base_name}_label.zarr"
            parent_directory = os.path.join(self.file_name, os.pardir)
            # Open the file dialog with the proposed file name
            save_file_path = filedialog.asksaveasfilename(
                initialdir=parent_directory,
                title="Select Directory to Save Mask Zarr",
                initialfile=default_save_file_name,
                filetypes=[("Zarr files", "*.zarr")]
            )

            if save_file_path:
                try:
                    # Save the Zarr array to the chosen file path
                    zarr.save_array(save_file_path, self.mask_data)
                    print(f"LOG: Mask saved as Zarr in {save_file_path}")
                except Exception as e:
                    print(f"LOG: Error saving mask as Zarr: {e}")
        else:
            print("LOG: No mask data to save.")

    def update_threshold_layer(self, layer):
        try:
            self.th_layer = int(float(layer))
            self.bucket_layer_var.set(f"{self.th_layer}")

            # Update the Bucket Threshold Slider to the current layer's threshold value
            current_threshold = self.threshold[self.th_layer]
            self.bucket_threshold_var.set(f"{current_threshold}")
            # You may need to adjust this line depending on how the slider is named in your code
            self.bucket_threshold_slider.set(current_threshold)

            print(f"LOG: Layer {self.th_layer} selected, current threshold is {current_threshold}.")
        except ValueError:
            print("LOG: Invalid layer value.")

    def update_threshold_value(self, val):
        try:
            self.threshold[self.th_layer] = int(float(val))
            self.bucket_threshold_var.set(f"{int(float(val))}")
            print(f"LOG: Layer {self.th_layer} threshold set to {self.threshold[self.th_layer]}.")
        except ValueError:
            print("LOG: Invalid threshold value.")

    def threaded_flood_fill(self):
        if self.click_coordinates and self.voxel_data is not None:
            # Run flood_fill_3d in a separate thread
            thread = threading.Thread(target=self.flood_fill_3d, args=(self.click_coordinates,))
            thread.start()
        else:
            print("LOG: No starting point or data for flood fill.")

    def flood_fill_3d(self, start_coord):
        self.flood_fill_active = True
        target_color = self.voxel_data[start_coord]
        queue = deque([start_coord])
        visited = set()

        counter = 0
        while self.flood_fill_active and queue and counter < self.max_propagation_steps:
            cz, cy, cx = queue.popleft()

            if (cz, cy, cx) in visited or not (0 <= cz < self.voxel_data.shape[0] and 0 <= cy < self.voxel_data.shape[1] and 0 <= cx < self.voxel_data.shape[2]):
                continue

            visited.add((cz, cy, cx))

            if self.barrier_mask[cz, cy, cx] != 0:
                continue

            if abs(int(self.voxel_data[cz, cy, cx]) - int(target_color)) <= self.threshold[cz]:
                self.mask_data[cz, cy, cx] = 1
                counter += 1
                for dz in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dz == 0 and dx == 0 and dy == 0:
                                continue
                            queue.append((cz + dz, cy + dy, cx + dx))

            if counter % 10 == 0:
                self.root.after(1, self.update_display_slice)
        if self.flood_fill_active == True:
            self.flood_fill_active = False
            print("LOG: Flood fill ended.")

    def stop_flood_fill(self):
        self.flood_fill_active = False
        print("LOG: Flood fill stopped.")

    def save_state(self):
        # Save the current state of the image before modifying it
        if self.voxel_data is not None:
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

    def on_canvas_drag(self, event):
        if self.drag_start_x is not None and self.drag_start_y is not None:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.image_position_x -= dx
            self.image_position_y -= dy
            self.image_position_x = min(max(self.image_position_x, 0), self.dataset.shape[0])
            self.image_position_y = min(max(self.image_position_y, 0), self.dataset.shape[1])
            self.update_display_slice()
            self.drag_start_x, self.drag_start_y = event.x, event.y

    def on_canvas_pencil_drag(self, event):
        if self.mode.get() == "pencil" or self.mode.get() == "eraser":
            self.save_state()
            self.color_pixel(self.calculate_image_coordinates(event))

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
        if canvas_width <= 1 or canvas_height <= 1:
            print("Canvas not yet initialized, can't update display slice")
            return

        needed_data_width, needed_data_height = math.ceil(canvas_width / self.zoom_level), math.ceil(canvas_height / self.zoom_level)
        x0, y0 = self.image_position_x, self.image_position_y
        # print('x0, y0:', x0, y0)

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

            img = Image.fromarray((self.voxel_data[:, :, self.z_index]).astype('uint16').swapaxes(0, 1)).convert('RGBA')
            # print('img shape', np.array(img).shape)
        else:
            img = Image.fromarray(np.zeros(shape=(1, needed_data_width, needed_data_height)).astype('uint16')).convert('RGBA')
        # print(np.array(img).shape)

        # # Only overlay the mask if show_mask is True
        # if self.mask_data is not None and self.show_mask:
        #     mask = np.uint8(self.mask_data[x0:x0+needed_data_width, y0:y0+needed_data_height] * self.overlay_alpha)
        #     yellow = np.zeros_like(mask, dtype=np.uint8)
        #     yellow[:, :] = 255  # Yellow color
        #     mask_img = Image.fromarray(np.stack([yellow, yellow, np.zeros_like(mask), mask], axis=-1).swapaxes(0, 1), 'RGBA')

        #     # Overlay the mask on the original image
        #     img = Image.alpha_composite(img, mask_img)

        # if self.barrier_mask is not None and self.show_barrier:
        #     barrier = np.uint8(self.barrier_mask[self.z_index, :, :] * self.overlay_alpha)
        #     red = np.zeros_like(barrier, dtype=np.uint8)
        #     red[:, :] = 255  # Red color
        #     barrier_img = Image.fromarray(np.stack([red, np.zeros_like(barrier), np.zeros_like(barrier), barrier], axis=-1), 'RGBA')

        #     # Overlay the barrier mask on the original image
        #     img = Image.alpha_composite(img, barrier_img)

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
        self.save_state()
        img_coords = self.calculate_image_coordinates(event)
        if self.mode.get() == "bucket":
            if self.flood_fill_active == True:
                print("LOG: Last flood fill hasn't finished yet.")
            else:
                # Assuming the flood fill functionality
                self.click_coordinates = img_coords
                print("LOG: Starting flood fill...")
                self.threaded_flood_fill()  # Assuming threaded_flood_fill is implemented for non-blocking UI
        elif self.mode.get() == "pencil":
            # Assuming the pencil (pixel editing) functionality
            self.color_pixel(img_coords)  # Assuming color_pixel is implemented

    def calculate_image_coordinates(self, input):
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
            pan_adjusted_x = x - self.image_position_x
            pan_adjusted_y = y - self.image_position_y

            # Calculate the position in the zoomed image
            zoomed_image_x = max(0, min(pan_adjusted_x, zoomed_width))
            zoomed_image_y = max(0, min(pan_adjusted_y, zoomed_height))

            # Scale back to original image coordinates
            img_x = int(zoomed_image_x / self.zoom_level)
            img_y = int(zoomed_image_y / self.zoom_level)

            # Debugging output
            #print(f"Clicked at: ({x}, {y}), Image Coords: ({img_x}, {img_y})")

            return self.z_index, img_y, img_x

    def color_pixel(self, img_coords):
        z_index, canvas_center_y, canvas_center_x = img_coords
        # Left top corner is at image_position_x, image_position_y, and we have the center in canvas coords
        center_x, center_y = canvas_center_x + self.image_position_x, canvas_center_y + self.image_position_y
        print('center:', center_x*4, center_y*4)

        if self.voxel_data is not None:
            # Calculate the square bounds of the circle
            min_x = max(0, center_x - self.pencil_size)
            max_x = min(self.voxel_data.shape[0] - 1, center_x + self.pencil_size)
            min_y = max(0, center_y - self.pencil_size)
            max_y = min(self.voxel_data.shape[1] - 1, center_y + self.pencil_size)

        if self.mode.get() in ["pencil", "eraser"]:
            # Decide which mask to edit based on editing_barrier flag
            target_mask = self.barrier_mask if self.editing_barrier else self.mask_data
            mask_value = 1 if self.mode.get() == "pencil" else 0
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Check if the pixel is within the circle's radius
                    if math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) <= self.pencil_size:
                        target_mask[x, y] = mask_value
            self.update_display_slice()


    def update_pencil_size(self, val):
        self.pencil_size = int(float(val))
        self.pencil_size_var.set(f"{self.pencil_size}")
        print(f"LOG: Pencil size set to {self.pencil_size}")

    def update_pencil_cursor(self, event):
        # Remove the old cursor representation
        if self.pencil_cursor:
            self.canvas.delete(self.pencil_cursor)
            self.update_display_slice()

        if self.mode.get() == "pencil":
            color = "yellow" if not self.editing_barrier else "red"
        if self.mode.get() == "eraser":
            color = "white"
        if self.mode.get() == "eraser" or self.mode.get() == "pencil":
            radius = self.pencil_size * self.zoom_level  # Adjust radius based on zoom level
            self.pencil_cursor = self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline=color, width=2)
        self.click_coordinates = (self.z_index, event.y, event.x)
        self.update_info_display()

    def scroll_or_zoom(self, event):
        # Adjust for different platforms
        ctrl_pressed = False
        if sys.platform.startswith('win'):
            # Windows
            ctrl_pressed = event.state & 0x0004
            delta = event.delta
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            # Linux or macOS
            ctrl_pressed = event.state & 4
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
        print(f"LOG: Label {'shown' if self.show_mask else 'hidden'}.\n")

    def toggle_barrier(self):
        # Toggle the state
        self.show_barrier = not self.show_barrier
        # Update the variable for the Checkbutton
        self.show_barrier_var.set(self.show_barrier)
        # Update the display to reflect the new state
        self.update_display_slice()
        print(f"LOG: Barrier {'shown' if self.show_barrier else 'hidden'}.\n")

    def toggle_image(self):
        # Toggle the state
        self.show_image = not self.show_image
        # Update the variable for the Checkbutton
        self.show_image_var.set(self.show_image)
        # Update the display to reflect the new state
        self.update_display_slice()
        print(f"LOG: Image {'shown' if self.show_image else 'hidden'}.\n")

    def toggle_editing_mode(self):
        # Toggle between editing label and barrier
        self.editing_barrier = not self.editing_barrier
        print(f"LOG: Editing {'Barrier' if self.editing_barrier else 'Label'}")

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

Commands Overview:
- Icons (Top, Left to Right):
  1. Open Zarr 3D Image: Load image data from a Zarr directory.
  2. Open Zarr 3D Label: Load label data from a Zarr directory.
  3. Save Zarr 3D Label: Save current label data to a Zarr file.
  4. Undo Last Action: Revert the last change made to the label or barrier.
  5. Brush Tool: Edit labels or barriers with a freehand brush.
  6. Eraser Tool: Erase parts of the label or barrier.
  7. Edit Barrier: Toggle between editing the label or the barrier mask.
  8. Pencil Size: Adjust the size of the brush and eraser tools.
  9. 3D Flood Fill Tool: Fill an area with the label based on similarity.
  10. STOP: Interrupt the ongoing flood fill operation.
  11. Info: Display information and usage tips.

- Sliders and Toggles (Bottom):
  1. Toggle Label: Show or hide the label overlay.
  2. Toggle Barrier: Show or hide the barrier overlay.
  3. Opacity: Adjust the transparency of the label and barrier overlays.
  4. Toggle Image: Show or hide the image data.
  5. Bucket Layer: Select the layer to adjust its specific flood fill threshold.
  6. Bucket Threshold: Set the threshold for the flood fill tool.
  7. Max Propagation: Limit the extent of the flood fill operation.

Usage Tips:
- Pouring Gold: The 3D flood fill algorithm labels contiguous areas based on voxel intensity and the set threshold.
    The gold does not propagate into the barrier.
- Navigation: Click and drag with the left mouse button to pan the image.
- Zoom: Use CTRL+Scroll to zoom in and out. Change the Z-axis slice with the mouse wheel.
- Editing Modes: Use the "Edit Barrier" toggle to switch between modifying the label and the barrier mask.
- Overlay Visibility: Use the toggle buttons to show or hide the label, barrier, and image data for easier editing.
- Tool Size: Use the "Pencil Size" slider to adjust the size of the brush and eraser.

Created by Dr. Giorgio Angelotti, Vesuvius Kintsugi is designed for efficient 3D voxel image labeling. Released under the MIT license.
"""
        # Insert the help text into the text widget and disable editing
        help_text_widget.insert("1.0", info_text)

    def update_max_propagation(self, val):
        self.max_propagation_steps = int(float(val))
        self.max_propagation_var.set(f"{self.max_propagation_steps}")
        print(f"LOG: Max Propagation Steps set to {self.max_propagation_steps}")

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

        save_button = ttk.Button(self.toolbar_frame, image=save_icon, command=self.save_image)
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

        self.editing_barrier_var = tk.BooleanVar(value=self.editing_barrier)
        toggle_editing_button = ttk.Checkbutton(self.toolbar_frame, text="Edit Barrier", command=self.toggle_editing_mode, variable=self.editing_barrier_var)
        toggle_editing_button.pack(side=tk.LEFT, padx=5)

        self.pencil_size_var = tk.StringVar(value="10")  # Default pencil size
        pencil_size_label = ttk.Label(self.toolbar_frame, text="Pencil Size:")
        pencil_size_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        pencil_size_slider = ttk.Scale(self.toolbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_pencil_size)
        pencil_size_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(pencil_size_slider, "Adjust Pencil Size")

        pencil_size_value_label = ttk.Label(self.toolbar_frame, textvariable=self.pencil_size_var)
        pencil_size_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Bucket tool button
        bucket_button = ttk.Radiobutton(self.toolbar_frame, image=bucket_icon, variable=self.mode, value="bucket")
        bucket_button.image = bucket_icon
        bucket_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(bucket_button, "Flood Fill Tool")

        # Stop tool button
        stop_button = ttk.Button(self.toolbar_frame, image=stop_icon, command=self.stop_flood_fill)
        stop_button.image = stop_icon
        stop_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(stop_button, "Stop Flood Fill")

        # Help button
        help_button = ttk.Button(self.toolbar_frame, image=help_icon, command=self.show_help)
        help_button.image = help_icon
        help_button.pack(side=tk.RIGHT, padx=2)
        self.create_tooltip(help_button, "Info")

        # Bucket Threshold Slider
        '''
        self.bucket_threshold_var = tk.StringVar(value="4")  # Default threshold
        bucket_threshold_label = ttk.Label(self.toolbar_frame, text="Bucket Threshold:")
        bucket_threshold_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        self.bucket_threshold_slider = ttk.Scale(self.toolbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_threshold_value)
        self.bucket_threshold_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.bucket_threshold_slider, "Adjust Bucket Threshold")

        bucket_threshold_value_label = ttk.Label(self.toolbar_frame, textvariable=self.bucket_threshold_var)
        bucket_threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))
        '''
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
        self.show_barrier_var = tk.BooleanVar(value=self.show_barrier)
        self.show_image_var = tk.BooleanVar(value=self.show_image)

        # Create a frame to hold the toggle buttons
        toggle_frame = tk.Frame(self.root)
        toggle_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

        # Create toggle buttons for mask and image visibility
        toggle_mask_button = ttk.Checkbutton(toggle_frame, text="Toggle Label", command=self.toggle_mask, variable=self.show_mask_var)
        toggle_mask_button.pack(side=tk.LEFT, padx=5, anchor='s')

        toggle_barrier_button = ttk.Checkbutton(toggle_frame, text="Toggle Barrier", command=self.toggle_barrier, variable=self.show_barrier_var)
        toggle_barrier_button.pack(side=tk.LEFT, padx=5, anchor='s')

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

        self.bucket_layer_slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_threshold_layer)
        self.bucket_layer_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.bucket_layer_slider, "Adjust Bucket Layer")

        bucket_layer_value_label = ttk.Label(slider_frame, textvariable=self.bucket_layer_var)
        bucket_layer_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Bucket Threshold Slider
        self.bucket_threshold_var = tk.StringVar(value="4")
        bucket_threshold_label = ttk.Label(slider_frame, text="Bucket Threshold:")
        bucket_threshold_label.pack(side=tk.LEFT, padx=(10, 2))

        self.bucket_threshold_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_threshold_value)
        self.bucket_threshold_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.bucket_threshold_slider, "Adjust Bucket Threshold")

        bucket_threshold_value_label = ttk.Label(slider_frame, textvariable=self.bucket_threshold_var)
        bucket_threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Max Propagation Slider
        self.max_propagation_var = tk.IntVar(value=self.max_propagation_steps)
        max_propagation_label = ttk.Label(slider_frame, text="Max Propagation:")
        max_propagation_label.pack(side=tk.LEFT, padx=(10, 2))

        max_propagation_slider = ttk.Scale(slider_frame, from_=1, to=500, orient=tk.HORIZONTAL, command=self.update_max_propagation)
        max_propagation_slider.set(self.max_propagation_steps)
        max_propagation_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(max_propagation_slider, "Adjust Max Propagation Steps for Flood Fill")

        max_propagation_value_label = ttk.Label(slider_frame, textvariable=self.max_propagation_var)
        max_propagation_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Create a frame for the log text area and scrollbar
        log_frame = tk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Update display canvas size, then load data from dataset and display it
        self.root.update_idletasks()
        self.update_display_slice()

        self.root.mainloop()

if __name__ == "__main__":
    editor = VesuviusKintsugi()
