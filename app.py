import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from datasets import load_dataset
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
dataset = load_dataset("nielsr/breast-cancer", split="train").train_test_split(test_size=0.2, seed=42)['test']

# Cargar el modelo y el procesador
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Cargar los pesos del modelo guardados
try:
    weights_path = 'sam_model_weights.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Pesos del modelo cargados desde {weights_path}")
except FileNotFoundError:
    print(f"Advertencia: no se encontró el archivo de pesos en {weights_path}. Usando pesos preentrenados.")

model.to(device)
model.eval()

def apply_mask(image_pil, mask_np):
    """Aplica una máscara de color a una imagen PIL."""
    if image_pil is None or mask_np is None:
        return Image.new('RGB', (256, 256), 'white')
    image_rgba = image_pil.convert("RGBA")
    # Color de la máscara: azul semitransparente
    color = np.array([30, 144, 255, 153], dtype=np.uint8)
    mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    mask_rgba[mask_np > 0] = color
    mask_image_pil = Image.fromarray(mask_rgba)
    
    # Superponer la máscara en la imagen original
    return Image.alpha_composite(image_rgba, mask_image_pil)

# Función principal de inferencia
def run_inference(image_pil, prompt_box):
    """Ejecuta la inferencia en la imagen seleccionada con un aviso de cuadro delimitador."""
    blank_image = Image.new('RGB', (256, 256), 'white')
    if image_pil is None:
        return blank_image, blank_image, blank_image, "Por favor, carga una imagen primero."
    if prompt_box is None:
        return image_pil, image_pil, image_pil, "Por favor, define un rectángulo con dos clics en la imagen."

    inputs = processor(image_pil, input_boxes=[[prompt_box]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    predicted_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    ground_truth_mask = None
    image_np = np.array(image_pil)
    for example in dataset:
        if np.array_equal(np.array(example['image']), image_np):
            ground_truth_mask = np.array(example['label'])
            break
    
    gt_image_with_mask = apply_mask(image_pil, ground_truth_mask)
    pred_image_with_mask = apply_mask(image_pil, predicted_mask)

    return image_pil, gt_image_with_mask, pred_image_with_mask, "Inferencia completada."

# Interfaz de Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Componentes de estado para mantener los datos entre interacciones
    image_state = gr.State()
    prompt_box_state = gr.State()
    # Nuevo estado para almacenar la primera esquina del rectángulo
    corner_one_state = gr.State()

    gr.Markdown("## Inferencia Interactiva de Cáncer de Mama con SAM")
    gr.Markdown("1. Elige una imagen del menú. 2. Haz clic dos veces en la imagen para definir un rectángulo. 3. Ejecuta la inferencia.")

    with gr.Row():
        # --- Panel Izquierdo ---
        with gr.Column(scale=2, min_width=300):
            gr.Markdown("<h3 style='text-align: center;'>Seleccionar Imagen de Prueba</h3>")
            image_selector = gr.Dropdown(
                choices=[f"Image {i}" for i in range(len(dataset))],
                show_label=False
            )
            gr.Markdown("<h3 style='text-align: center;'>Imagen para Aviso Interactivo</h3>")
            input_image = gr.Image(type="pil", sources=[], show_label=False)
            run_button = gr.Button("Ejecutar Inferencia")
            gr.Markdown("<h3 style='text-align: center;'>Estado</h3>")
            status_text = gr.Textbox(interactive=False, lines=2, show_label=False)

        # --- Panel Derecho ---
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("<h3 style='text-align: center;'>Imagen Original</h3>")
                    out_original = gr.Image(type="pil", show_label=False)
                with gr.Column():
                    gr.Markdown("<h3 style='text-align: center;'>Verdad Fundamental</h3>")
                    out_gt = gr.Image(type="pil", show_label=False)
                with gr.Column():
                    gr.Markdown("<h3 style='text-align: center;'>Máscara Predicha</h3>")
                    out_pred = gr.Image(type="pil", show_label=False)
    
    # --- Funciones para los eventos de la UI ---
    def load_image_from_selector(selection_str):
        """Carga la imagen seleccionada y reinicia los estados."""
        if not selection_str:
            return None, None, None, None, "Por favor, selecciona una imagen."
        image_idx = int(selection_str.split(' ')[1])
        image_pil = dataset[image_idx]["image"]
        # Reiniciar todos los estados al cargar una nueva imagen
        return image_pil, image_pil, None, None, "Imagen cargada. Haz clic una vez para la primera esquina."
    
    def handle_image_click(first_corner, evt: gr.SelectData):
        """Maneja los clics para definir el cuadro delimitador."""
        clicked_point = evt.index
        if first_corner is None:
            # Es el primer clic, guardamos la esquina
            return clicked_point, None, f"Primera esquina en {clicked_point}. Haz clic de nuevo para la segunda."
        else:
            # Es el segundo clic, creamos la caja
            x1, y1 = first_corner
            x2, y2 = clicked_point
            prompt_box = [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))]
            # Reiniciamos el estado de la primera esquina para la próxima vez
            return None, prompt_box, f"Rectángulo definido. Listo para la inferencia."

    # --- Conectar los componentes de la UI a las funciones ---
    image_selector.change(
        fn=load_image_from_selector,
        inputs=image_selector,
        outputs=[input_image, image_state, prompt_box_state, corner_one_state, status_text]
    )

    input_image.select(
        fn=handle_image_click,
        inputs=[corner_one_state],
        outputs=[corner_one_state, prompt_box_state, status_text]
    )

    run_button.click(
        fn=run_inference,
        inputs=[image_state, prompt_box_state],
        outputs=[out_original, out_gt, out_pred, status_text]
    )

demo.launch(debug=True)
