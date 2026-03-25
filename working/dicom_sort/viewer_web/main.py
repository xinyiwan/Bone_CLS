import streamlit as st
import os
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from PIL import Image
from utils import pad_image, load_dicom_dataframe

st.set_page_config(layout="wide", page_title="DICOM Classifier")

n_cols = 4
n_rows = 3

def obtener_clase_inicial(w, fs, c):
    """Genera la etiqueta de clase por defecto basada en las columnas del Excel."""
    w, fs, c = str(w).strip(), str(fs).strip(), str(c).strip()
    
    if w == "T1W":
        str_fs = "FS" if fs == "Y" else "noFS"
        str_c = "CE" if c == "Y" else "noCE"
        return f"T1W {str_fs} {str_c}"
    elif w == "T2W":
        str_fs = "FS" if fs == "Y" else "noFS"
        return f"T2W {str_fs}"
    elif w in ["DW"]:
        return "DW"
    elif w == "Other":
        return "Other"
    return "To_review"

def decodificar_seleccion(seleccion, orig_w, orig_fs, orig_c):
    """Convierte la pastilla seleccionada de nuevo a las 3 variables individuales."""
    mapeo = {
        "T1W noFS noCE": ("T1W", "N", "N"),
        "T1W FS noCE": ("T1W", "Y", "N"),
        "T1W noFS CE": ("T1W", "N", "Y"),
        "T1W FS CE": ("T1W", "Y", "Y"),
        "T2W noFS": ("T2W", "N", "-"),
        "T2W FS": ("T2W", "Y", "-"),
        "DW": ("DW", "-", "-"),
        "Other": ("Other", "-", "-"),
        "To_review": ("To_review", "To_review", "To_review")
    }
    return mapeo.get(seleccion, (orig_w, orig_fs, orig_c))

def ejecutar_script_generacion(ruta_excel, ruta_img_base):
    """Ejecuta un script externo para generar las imágenes."""
    # Sustituye 'generar_imagenes.py' por el nombre real de tu script
    script_path = "img_generator.py" 
    
    if not os.path.exists(script_path):
        st.error(f"No se encontró el script de generación: {script_path}")
        return False

    try:
        # Llamamos al script pasando como argumentos el excel y la ruta base
        st.info("Ejecutando script de generación... Por favor, espera.")
        resultado = subprocess.run(
            ["python3", script_path, "--excel", ruta_excel, "--out_dir", ruta_img_base],
            capture_output=True, text=True, check=True
        )
        st.success("¡Imágenes generadas con éxito!")
        # Opcional: st.code(resultado.stdout) para ver el log del script
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error al ejecutar el script:\n{e.stderr}")
        return False
    except Exception as e:
        st.error(f"Error inesperado:\n{e}")
        return False
    

def main():
    
    st.sidebar.title("Configuración de Carga")
    
    path_results = st.sidebar.text_input("1. Ruta guardado revisión", value="/Proyecto/Results")
    path_img_base = st.sidebar.text_input("2. Ruta base de imágenes", value="/Proyecto/IMG")
    uploaded_file = st.sidebar.file_uploader("3. Sube el Excel original", type=["xlsx"])


    if uploaded_file is None:
        st.title("🩻 Clasificador DICOM")
        st.info("Esperando archivo Excel... Por favor, súbelo desde la barra lateral.")
        return
    
    # --- BOTÓN PARA GENERAR IMÁGENES ---
    st.sidebar.divider()
    if st.sidebar.button("🛠️ Generar imágenes faltantes", help="Ejecuta un script externo para crear las imágenes .png a partir del Excel subido."):
        # Necesitamos la ruta real del archivo excel subido.
        # En Streamlit, uploaded_file está en memoria, así que lo guardamos temporalmente.
        temp_excel_path = os.path.join(path_results, "temp_upload.xlsx")
        os.makedirs(path_results, exist_ok=True)
        with open(temp_excel_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if ejecutar_script_generacion(temp_excel_path, path_img_base):
            st.toast("Proceso de generación finalizado.")
            
        # Opcional: borrar el archivo temporal
        if os.path.exists(temp_excel_path):
            os.remove(temp_excel_path)

    # --- LÓGICA DE DETECCIÓN DE PROGRESO PREVIO ---
    # Creamos el nombre del archivo maestro actualizado basándonos en el original
    master_filename = f"Review_{uploaded_file.name}"
    master_filepath = os.path.join(path_results, master_filename)
    
    usar_progreso = False
    if os.path.exists(master_filepath):
        st.sidebar.warning(f"⚠️ Se ha detectado un progreso previo:\n**{master_filename}**")
        usar_progreso = st.sidebar.checkbox("Cargar este progreso", value=True, help="Desmarca esto si quieres empezar de cero con el Excel original.")

    # --- LÓGICA DE CARGA DE DATOS ---
    if 'df_master' not in st.session_state or st.sidebar.button("🔄 Cargar / Recargar"):
        if usar_progreso and os.path.exists(master_filepath):
            # Cargar el archivo que ya tiene los avances
            df_temp = pd.read_excel(master_filepath)
            st.toast("Progreso anterior cargado con éxito", icon="📁")
        else:  
            df_temp = load_dicom_dataframe(uploaded_file)

            if "viewed" not in df_temp.columns:
                df_temp["viewed"] = ""
            if "Clase W Final" not in df_temp.columns:
                df_temp["Clase W Final"] = df_temp["Predicción Clases W"]
            if "Clase FS Final" not in df_temp.columns:
                df_temp["Clase FS Final"] = df_temp["Predicción Clases FS"]
            if "Clase C Final" not in df_temp.columns:
                df_temp["Clase C Final"] = df_temp["Predicción Clases C"]
                
            st.toast("Excel original cargado desde cero", icon="📄")
            
        st.session_state.df_master = df_temp
        st.session_state.master_filepath = master_filepath

    df = st.session_state.df_master
    
    # --- TABLA RESUMEN ---
    with st.expander("📊 Ver Tabla Resumen de Casos", expanded=False):
        resumen_df = df.groupby(["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"]).size().reset_index(name="Total")
        revisados_df = df[df["viewed"] == "X"].groupby(["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"]).size().reset_index(name="Revisados")
        
        tabla_final = pd.merge(resumen_df, revisados_df, on=["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"], how="left").fillna(0)
        tabla_final["Revisados"] = tabla_final["Revisados"].astype(int)
        tabla_final["Pendientes"] = tabla_final["Total"] - tabla_final["Revisados"]
        
        st.dataframe(tabla_final, use_container_width=True, hide_index=True)

    # --- FILTROS CONDICIONALES ---
    st.sidebar.divider()
    st.sidebar.subheader("Filtros de Secuencia")
    
    val_w = st.sidebar.selectbox("Clase W", ["T1W", "T2W", "DW", "Other"])
    mask = (df["Predicción Clases W"] == val_w)

    if val_w in ["DW", "Other"]:
        val_fs = "-"
    else:
        val_fs = st.sidebar.selectbox("Clase FS", ["N", "Y"])
        mask = mask & (df["Predicción Clases FS"] == val_fs)

    if val_w != "T1W":
        val_c = "-"
    else:
        val_c = st.sidebar.selectbox("Clase C", ["N", "Y"])
        mask = mask & (df["Predicción Clases C"] == val_c)

    show_only_pending = st.sidebar.checkbox("Mostrar solo pendientes", value=True)
    if show_only_pending:
        mask = mask & (df["viewed"] != "X")

    df_filtered = df[mask].reset_index()

    # --- INTERFAZ PRINCIPAL ---
    st.title(f"Revisión: {val_w} | FS: {val_fs} | CE: {val_c}")
    
    if df_filtered.empty:
        st.success("✅ ¡Todo revisado para este filtro o no existen casos!")
        return


    batch = df_filtered.head(n_cols*n_rows)  
    button_labels = ["T1W noFS noCE", "T1W FS noCE", "T1W noFS CE", "T1W FS CE", "T2W noFS", "T2W FS", "DW", "Other", "To_review"]

    user_actions = {}

    with st.form("batch_form"):
        cols = st.columns(n_cols)
        for i, (f_idx, row) in enumerate(batch.iterrows()):
            orig_idx = row['index']
            with cols[i % n_cols]:
                paciente = str(row.get("Paciente", ""))
                estudio = str(row.get("Estudio", ""))
                serie = str(row.get("Serie", ""))
                
                full_path = os.path.join(path_img_base, paciente, estudio, serie, "Img.png")
                
                try:
                    img_pil = Image.open(full_path).convert('L')
                    img_array = np.array(img_pil)
                    img_padded = pad_image(img_array)
                    st.image(img_padded, use_container_width=True, clamp=True)
                except Exception as e: 
                    st.error(f"Img no encontrada o error al procesar:\n{serie[:10]}...\nDetalle del error: {e}")

                orig_w = row.get("Predicción Clases W", "")
                orig_fs = row.get("Predicción Clases FS", "")
                orig_c = row.get("Predicción Clases C", "")
                
                clase_por_defecto = obtener_clase_inicial(orig_w, orig_fs, orig_c)
                
                res = st.pills("Clase", button_labels, default=clase_por_defecto, key=f"p_{orig_idx}", label_visibility="collapsed")
                user_actions[orig_idx] = res

        # --- GUARDADO DUAL ---
        if st.form_submit_button("💾 Guardar y Siguiente", use_container_width=True):
            for idx, seleccion_final in user_actions.items():
                fila_original = st.session_state.df_master.loc[idx]
                orig_w = fila_original.get("Predicción Clases W", "")
                orig_fs = fila_original.get("Predicción Clases FS", "")
                orig_c = fila_original.get("Predicción Clases C", "")
                
                if not seleccion_final:
                    seleccion_final = obtener_clase_inicial(orig_w, orig_fs, orig_c)
                
                w_final, fs_final, c_final = decodificar_seleccion(seleccion_final, orig_w, orig_fs, orig_c)
                
                #st.session_state.df_master.at[idx, "seleccion"] = seleccion_final
                st.session_state.df_master.at[idx, "viewed"] = "X"
                st.session_state.df_master.at[idx, "Clase W Final"] = w_final
                st.session_state.df_master.at[idx, "Clase FS Final"] = fs_final
                st.session_state.df_master.at[idx, "Clase C Final"] = c_final
                
            os.makedirs(path_results, exist_ok=True)
            
            # 1. Guardar copia maestra (se sobreescribe constantemente con los nuevos avances)
            st.session_state.df_master.to_excel(st.session_state.master_filepath, index=False)
            
            # 2. Guardar archivo de trazabilidad en subcarpeta "Reviews"
            path_reviews = os.path.join(path_results, "Reviews")
            os.makedirs(path_reviews, exist_ok=True)
            output_name_traz = os.path.join(path_reviews, f"Revision_{val_w}_{datetime.now().strftime('%Y%m%d-%H%M')}.xlsx")
            st.session_state.df_master.to_excel(output_name_traz, index=False)
            
            st.rerun()

if __name__ == "__main__":
    main()