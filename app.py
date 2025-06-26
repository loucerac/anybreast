import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from transformers import SamProcessor, SamModel
import gradio as gr
import torch.nn as nn # Asegúrate de importar nn para DataParallel si no lo hiciste antes

# --- 1. Cargar el conjunto de datos y el modelo (adaptado de tu script original) ---
# Si ya has ejecutado las celdas anteriores de tu notebook, puedes saltar la carga del modelo.
# Pero para que este script sea autocontenido, lo incluimos aquí.

print("Cargando conjunto de datos...")
dataset_orig = load_dataset("nielsr/breast-cancer", split="train")
dataset = dataset_orig.train_test_split(test_size=2, seed=42) # Mantener el test_size pequeño para la demo
print("Conjunto de datos cargado.")

print("Cargando modelo SAM...")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

print("Cargando pesos SAM...")
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.load_state_dict(torch.load('sam_model_weights.pth'))


# Asegúrate de que el modelo esté en el dispositivo correcto y sea DataParallel si es necesario
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.DataParallel(model) # Si usaste DataParallel en el entrenamiento
model.to(device)
model.eval() # Asegúrate de que el modelo esté en modo evaluación para la inferencia
print(f"Modelo SAM cargado y en modo evaluación en dispositivo: {device}.")

# --- 2. Funciones auxiliares para visualización y el prompt de punto ---

def show_mask(mask, ax, color=[30/255, 144/255, 255/255, 0.6]): # Color fijo para las máscaras
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# --- 3. Función principal de inferencia interactiva ---

def run_sam_inference(image_index, selected_point_x, selected_point_y):
    """
    Ejecuta la inferencia del modelo SAM basada en una imagen seleccionada y un punto.

    Args:
        image_index (int): El índice de la imagen en el conjunto de datos de prueba.
        selected_point_x (float): La coordenada X del clic del usuario.
        selected_point_y (float): La coordenada Y del clic del usuario.

    Returns:
        tuple: Un tuple de 3 imágenes PIL: (imagen_original, imagen_con_gt_mask, imagen_con_pred_mask)
    """
    if selected_point_x is None or selected_point_y is None:
        # Si no se ha hecho clic, mostramos solo la imagen original y la verdad fundamental.
        # Esto puede ocurrir si el usuario cambia la imagen sin hacer clic.
        image = dataset["test"][image_index]["image"]
        ground_truth_mask = np.array(dataset["test"][image_index]["label"])

        fig_gt, axes_gt = plt.subplots(figsize=(6, 6))
        axes_gt.imshow(np.array(image))
        show_mask(ground_truth_mask, axes_gt)
        axes_gt.set_title("Máscara de Verdad Fundamental")
        axes_gt.axis("off")
        fig_gt.canvas.draw()
        gt_image_pil = Image.frombytes('RGB', fig_gt.canvas.get_width_height(), fig_gt.canvas.tostring_rgb())
        plt.close(fig_gt) # Cierra la figura para liberar memoria

        return image, gt_image_pil, Image.new('RGB', image.size, (128, 128, 128)) # Devuelve una imagen gris para la predicha

    # Cargar la imagen y la máscara de verdad fundamental
    image = dataset["test"][image_index]["image"]
    ground_truth_mask = np.array(dataset["test"][image_index]["label"])

    # Crear el prompt de punto a partir de las coordenadas del clic
    input_points = np.array([[selected_point_x, selected_point_y]])
    input_labels = np.array([1]) # Etiqueta 1 para un punto positivo

    # Preparar la imagen y el prompt para el modelo
    # El processor espera input_points como una lista de arrays de numpy
    # y input_labels como una lista de arrays de numpy
    inputs = processor(image, input_points=[input_points], input_labels=[input_labels], return_tensors="pt").to(device)

    # Realizar la inferencia
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # Aplicar sigmoide y convertir a máscara binaria
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

    # --- Generar visualizaciones ---
    # Imagen Original
    original_image_pil = image

    # Imagen con Máscara de Verdad Fundamental
    fig_gt, axes_gt = plt.subplots(figsize=(6, 6))
    axes_gt.imshow(np.array(image))
    show_mask(ground_truth_mask, axes_gt)
    axes_gt.set_title("Máscara de Verdad Fundamental")
    axes_gt.axis("off")
    fig_gt.canvas.draw()
    gt_image_pil = Image.frombytes('RGB', fig_gt.canvas.get_width_height(), fig_gt.canvas.tostring_rgb())
    plt.close(fig_gt)

    # Imagen con Máscara Predicha y el punto de prompt
    fig_pred, axes_pred = plt.subplots(figsize=(6, 6))
    axes_pred.imshow(np.array(image))
    show_mask(medsam_seg, axes_pred)
    show_points(input_points, input_labels, axes_pred) # Mostrar el punto usado como prompt
    axes_pred.set_title("Máscara Predicha (con Prompt)")
    axes_pred.axis("off")
    fig_pred.canvas.draw()
    pred_image_pil = Image.frombytes('RGB', fig_pred.canvas.get_width_height(), fig_pred.canvas.tostring_rgb())
    plt.close(fig_pred)

    return original_image_pil, gt_image_pil, pred_image_pil

# --- 4. Interfaz Gradio ---

# Creamos una lista de opciones para el selector de imagen
image_choices = [(f"Imagen {i}", i) for i in range(len(dataset["test"]))]

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Demo interactiva de SAM para Cáncer de Mama
        Selecciona una imagen, haz clic en una región de interés para generar un "prompt" y observa la segmentación predicha.
        """
    )

    with gr.Row():
        with gr.Column():
            image_selector = gr.Dropdown(
                choices=image_choices,
                label="Seleccionar Imagen de Prueba",
                value=0 # Por defecto selecciona la primera imagen
            )
            # Componente para mostrar la imagen seleccionada y capturar el clic
            input_image_display = gr.Image(
                label="Haz clic para un prompt",
                interactive=True,
                sources=["upload"] # Permitir carga para visualización si es necesario
            )
            run_button = gr.Button("Ejecutar Inferencia")

        with gr.Column():
            gr.Markdown("### Resultado de la Segmentación")
            with gr.Row():
                output_original = gr.Image(label="Imagen Original", interactive=False)
                output_ground_truth = gr.Image(label="Máscara Verdad Fundamental", interactive=False)
                output_predicted = gr.Image(label="Máscara Predicha", interactive=False)

    # Variables para almacenar el estado del clic
    clicked_x = gr.State(None)
    clicked_y = gr.State(None)

    # --- Lógica de la interfaz ---

    def update_image_display(image_idx):
        # Actualiza la imagen mostrada cuando se selecciona una nueva imagen
        image = dataset["test"][image_idx]["image"]
        # Limpiamos las coordenadas del clic anterior al cambiar de imagen
        return image, None, None

    image_selector.change(
        fn=update_image_display,
        inputs=image_selector,
        outputs=[input_image_display, clicked_x, clicked_y]
    )

    def store_click_coords(evt: gr.SelectData):
        # Almacena las coordenadas del clic cuando el usuario hace clic en la imagen
        return evt.index[1], evt.index[0]

    input_image_display.select(
        fn=store_click_coords,
        inputs=None,
        outputs=[clicked_x, clicked_y] # Gradio devuelve (y, x) para clics en imágenes
    )

    def trigger_inference(image_idx, x, y):
        # Esta función se llama cuando se pulsa el botón "Ejecutar Inferencia"
        # y pasa los valores del estado a la función de inferencia.
        return run_sam_inference(image_idx, x, y)


    run_button.click(
        fn=trigger_inference,
        inputs=[image_selector, clicked_x, clicked_y],
        outputs=[output_original, output_ground_truth, output_predicted]
    )

    # Cargar la primera imagen al iniciar la demo
    demo.load(
        fn=lambda: update_image_display(0), # Carga la imagen 0 por defecto
        inputs=None,
        outputs=[input_image_display, clicked_x, clicked_y]
    )


demo.launch(share=True)