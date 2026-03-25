import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.transform import resize
import numpy as np
from datetime import datetime


# Función para cargar un batch de imágenes
def load_batch(start_index):
    return df_to_analyse.iloc[start_index:start_index + batch_size]


# Función para manejar la selección desde los botones de Tkinter
def on_button_click(label, dcm_path):
    global selections
    selections[dcm_path] = label

def save_to_excel():
    df_output = df_filtered.copy()
    df_output["seleccion"] = df_output["Nombre DICOM"].apply(lambda x: selections.get(path_img + x[6:], ""))
    df_output["viewed"] = df_output["Nombre DICOM"].apply(lambda x: "X" if path_img + x[6:] in viewed_images else "")
    try:
        df_output["seleccion"] = df_filtered["seleccion"].where(df_filtered["seleccion"] == "X", df_output["seleccion"])
        df_output["viewed"] = df_filtered["viewed"].where(df_filtered["viewed"] == "X", df_output["viewed"])
    except:
        pass
    df_output.to_excel(output_excel, index=False)
    df_output.to_excel(output_excel_traz, index=False)
    print("Guardado en", output_excel)



# Función para manejar los clics sobre las imágenes
def on_click(event):
    if event.inaxes in axes_list:
        idx = axes_list.tolist().index(event.inaxes)
        row = df_to_analyse.iloc[current_index + idx]
        dcm_path = path_img + row["Nombre DICOM"][5:]
        print(f"Imagen seleccionada: {dcm_path}")

# Conectar el evento de clic


# Función para avanzar al siguiente batch
def next_batch():
    global current_index
    save_to_excel()
    if current_index + batch_size < len(df_to_analyse):
        current_index += batch_size
        update_view()
    else:
        # Desactivar el botón y cerrar la aplicación si no hay más batches
        next_button.config(state=tk.DISABLED)
        save_button.config(state=tk.DISABLED)
        print("No hay más batches. La aplicación se cerrará en breve.")
        root.after(2000, root.quit)  # Cerrar después de 2 segundos

def pad_image(img, target_height, target_width):
    height, width = img.shape
    scale_factor_h = target_height / height
    scale_factor_w= target_width / width
    if scale_factor_h > scale_factor_w:
        scale_factor= scale_factor_w
    else:
        scale_factor= scale_factor_h
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    img = resize(img, (new_height, new_width), anti_aliasing=True)
    pad_height = max(0, target_height - new_height)
    pad_width = max(0, target_width - new_width)
    padded_img = np.pad(img, ((pad_height // 2, pad_height - pad_height // 2),
                              (pad_width // 2, pad_width - pad_width // 2)),
                         mode="constant", constant_values=0)
    return padded_img[:target_height , :target_width]


def update_view():
    global current_index, axes_list, button_containers

    # Eliminar los button_frame anteriores
    for container in button_containers:
        container.destroy()
    button_containers = []  # Vaciar la lista después de eliminarlos

    batch = load_batch(current_index)
    fig, axes = plt.subplots(n_row, n_columns, figsize=(screen_width, screen_height))
    fig.suptitle(f"{sequence_name} {current_index // batch_size + 1} / {len(df_to_analyse) // batch_size + 1}", color="white", fontsize=12)
    fig.patch.set_facecolor("black")
    axes = axes.flatten()
    axes_list = axes

    # Calcular el tamaño objetivo
    target_width= int(root.winfo_screenwidth() /n_columns )
    target_height = int(root.winfo_screenheight() /n_row )
    valid_images = 0
    resol=root.winfo_screenwidth()/root.winfo_screenheight()
    if resol>1.7 and  resol<1.8:
        mov_y=0.082*root.winfo_screenheight()
        fact_x=0.4
    elif resol>1.55 and  resol<1.61:
        mov_y=0.092*root.winfo_screenheight()
        fact_x=0.4
    elif resol > 1.3 and resol < 1.4:
        mov_y=0.085*root.winfo_screenheight()
        fact_x=0.3
    else:
        mov_y=0.089*root.winfo_screenheight()
        fact_x=0.35

    for i, (idx, row) in enumerate(batch.iterrows()):
        name = row["Nombre DICOM"][6:]
        try:
            num_img=row["num_img"]
        except:
            num_img=0
        dcm_path = path_img+name
        folder_name = os.path.basename(os.path.dirname(dcm_path))
        viewed_images.add(dcm_path)

        try:
            ds = pydicom.dcmread(dcm_path)
            img = ds.pixel_array
            padded_img = pad_image(img, target_height, target_width)

            axes[i].imshow(padded_img, cmap="gray")
            axes[i].axis("off")
            axes[i].set_facecolor("black")
            valid_images += 1

            # Posición x
            screen_width_px = root.winfo_screenwidth()
            col = i % n_columns
            x_pos = int(screen_width_px / (n_columns ) * (col )+target_width*fact_x)
            x_pos = int(screen_width_px / (n_columns ) * (col ))

            # Posición y
            row_idx = i // n_columns
            fig_height_px = canvas.figure.bbox.height
            y_pos = int((fig_height_px/(n_row ))*(row_idx)*0.9 + mov_y)

            # Crear menú desplegable para seleccionar la secuencia
            button_frame = tk.Frame(root, bg="black")
            # Posicionamiento del botón
            button_frame.place(x=int(x_pos), y=int(y_pos))

            selected_option = tk.StringVar()
            selected_option.set("Choose")

            if num_img<10:
                label_img = tk.Label(button_frame, text=f"Img: {num_img}", fg="red", bg="black", font=("Arial", 10))
            else:
                label_img = tk.Label(button_frame, text=f"Img: {num_img}", fg="white",bg="black",font=("Arial", 10))

            label_img.pack(side="right", padx=20)

            def on_select(selection, path=dcm_path):
                on_button_click(selection, path)

            dropdown = tk.OptionMenu(button_frame, selected_option, *button_labels, command=on_select)
            dropdown.config(bg="gray", fg="white")
            dropdown.pack()

            menu = dropdown["menu"]
            color_mapping = {
                "T1W   noFS  noCE": "black",
                "T1W     FS    noCE": "black",
                "T1W   noFS     CE": "red",
                "T1W   FS     CE": "red",
                "T2W   noFS": "green",
                "T2W   FS": "green",
                "DWI": "black",
                "OTHERS": "black",
                "To_review": "black"
            }

            for idx, label in enumerate(button_labels):
                color = color_mapping.get(label, "black")
                menu.entryconfigure(idx, foreground=color)

            button_containers.append(button_frame)  # Guardar el frame en la lista
            axes[i].set_title(folder_name, fontsize=10, color="white", loc="center")

        except Exception as e:
            print(f"Error al cargar {dcm_path}: {e}")

    # Rellenar los espacios vacíos si no hay suficientes imágenes
    for i in range(len(batch), len(axes)):
        blank_img = np.zeros((target_height, target_width))
        axes[i].imshow(blank_img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_facecolor("black")

    for i in range(valid_images, len(axes)):
        axes[i].axis("off")
        axes[i].set_facecolor("black")

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0)
    canvas.figure = fig
    canvas.draw()



excel_path = "Z:\mnt\\rimp\PROJECTS\PRIMAGE\quibim-repository\propuesta_primagedisk\Code\Visualizacion\Sequence_Classifier_test_to_review.xlsx"
output_folder = "Z:/mnt/rimp/PROJECTS/PRIMAGE/quibim-repository/propuesta_primagedisk/NB/Classifier_review"
path_img = "Z:/mnt/rimp/PROJECTS/PRIMAGE/quibim-repository/propuesta_primagedisk/NB/"
secuencia_colum = ["Predicción Clases W","Predicción Clases FS","Predicción Clases C"]
sequence_filter = ["T1W","N","Y"]

sequence_name=sequence_filter[0]
for ind,name_seq in enumerate(sequence_filter):
    if ind == 1:
        if name_seq == "Y":
            sequence_name=sequence_name+'_FS'
        else:
            sequence_name=sequence_name+'_nFS'
    elif ind == 2:
        if name_seq == "Y":
            sequence_name=sequence_name+'_CE'
        else:
            sequence_name=sequence_name+'_nCE'

n_row=3
n_columns=8
batch_size = n_row*n_columns
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#output_folder = "Y:/mnt/rimp/PROJECTS/PRIMAGE/quibim-repository/propuesta_primagedisk/Seq_class/PRIMAGE/Subset/"
output_excel = os.path.join(output_folder,"Resultados_revision_"+sequence_name+".xlsx")
output_folder_trazability=os.path.join(output_folder,"Backup")
os.makedirs(output_folder_trazability, exist_ok=True)
output_excel_traz = os.path.join(output_folder_trazability,f"Review_{sequence_name}_{timestamp}.xlsx")

button_containers=[]
button_labels = ["T1W   noFS  noCE","T1W     FS    noCE","T1W   noFS     CE","T1W   FS     CE","T2W   noFS","T2W   FS","DWI", "OTHERS", "To_review"]
selections = {}
# Cargar el archivo Excel
try:
    df = pd.read_excel(output_excel_traz)
except:
    df = pd.read_excel(excel_path)


df_filtered=df.copy()
# Filtrar imágenes por secuencia seleccionada
for index,col in enumerate(secuencia_colum):
    df_filtered = df_filtered[df_filtered[col] == sequence_filter[index]].reset_index(drop=True)
try:
    df_to_analyse= df_filtered[df_filtered["viewed"] !="X"].reset_index(drop=True)
except:
    df_to_analyse = df_filtered

# Diccionario para almacenar selecciones
viewed_images= set()

current_index = 0

# Crear la ventana principal
root = tk.Tk()
root.title("DICOM image viewer")
root.configure(bg="black")  # Fondo negro en la ventana
root.attributes("-fullscreen", True)  # Ajustar a pantalla completa

frame = tk.Frame(root, bg="black")
frame.pack()

# Crear frame para los botones fuera de la función
button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=20)

screen_width = root.winfo_screenwidth()/100
screen_height = root.winfo_screenheight()/100*0.9

canvas = FigureCanvasTkAgg(plt.figure(figsize=(screen_width, screen_height)), master=frame)
canvas.get_tk_widget().pack()
canvas.mpl_connect("button_press_event", on_click)  # Conectar clics

# Botón siguiente batch
next_button = tk.Button(root, text="Next Batch", command=next_batch, fg="white", bg="black")
next_button.pack()

# Botón guardar y salir
save_button = tk.Button(root, text="Save and exit", command=lambda: [save_to_excel(), root.quit()], fg="white",
                        bg="black")
save_button.pack()

# Actualizar la vista inicial
update_view()
root.mainloop()
