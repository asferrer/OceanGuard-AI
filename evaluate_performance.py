"""
evaluate_performance.py

Script for quantitatively evaluating the performance of the OceanGuard AI model.
It runs both 'detection' and 'classification' evaluations in a single pass
for a comprehensive analysis.

Execution:
- Standard: python evaluate_performance.py
- Add -v for verbose output, --draw for visual comparisons, or --report for a visual summary.
"""
import torch
from unsloth import FastLanguageModel
from PIL import Image, ImageDraw, ImageFont
import json
import os
import re
import numpy as np
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

# --- CONFIGURACIÓN ---
IMAGE_DIR = "../RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/cleansea_dataset/CocoFormatDataset/train_coco/test_set"
ANNOTATION_FILE = "../RT-DETRv2-Densea/rtdetrv2_pytorch/dataset/cleansea_dataset/CocoFormatDataset/train_coco/annotations_densea_grouped.json"
VISUALS_DIR = "evaluation_visuals"
REPORT_FILE = "evaluation_report.png"

IOU_THRESHOLD = 0.5
LOG_FILE = "evaluation_log.txt"
LOG_INTERVAL = 10

# --- CARGA DEL MODELO ---

@torch.no_grad()
def setup_model():
    """Loads and configures the Gemma 3n model and tokenizer."""
    print("🚀 Loading Gemma 3n model for evaluation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3n-e2b-it",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': 0}
    )
    return model, tokenizer

# --- PROMPTS ESPECIALIZADOS ---

def get_detection_prompt(valid_classes):
    """Returns the prompt specialized for the detection task with high precision demands."""
    class_list_str = ", ".join([f"'{cls}'" for cls in valid_classes])
    return (
        "You are a precise, pixel-perfect marine debris detection system. "
        "Analyze the image and identify ALL man-made waste items. "
        "Return a JSON list of objects. Each object must have:\n"
        f"1. 'debris_type': The specific type of object. It MUST be one of the following: [{class_list_str}].\n"
        "2. 'material': The likely material (e.g., 'Plastic', 'Metal').\n"
        "3. 'confidence_score': A float from 0.0 to 1.0 indicating your certainty.\n"
        "4. 'bounding_box': A list of four normalized coordinates [x_min, y_min, x_max, y_max]. "
        "CRITICAL: The bounding box must be extremely precise, tightly enclosing the object with no extra padding. "
        "Do not include background, water, or shadows. Aim for pixel-perfect accuracy.\n"
        "Your response MUST be ONLY the JSON list."
    )

def get_classification_prompt(valid_classes):
    """Returns the prompt specialized for the classification task."""
    class_list_str = ", ".join([f"'{cls}'" for cls in valid_classes])
    return (
        "You are an expert marine debris classifier. "
        "Analyze the image and identify all types of man-made waste present. "
        "Return a JSON list containing the names of the debris types you see. "
        f"Each name MUST be one of the following: [{class_list_str}].\n"
        "Example: [\"Bottle\", \"Fishing_Net\", \"Can\"]\n"
        "Your response MUST be ONLY the JSON list."
    )

@torch.no_grad()
def run_inference(model, tokenizer, image_path, prompt_text):
    """Runs inference on a single image with a given prompt."""
    raw_response = "Error during image processing."
    try:
        image = Image.open(image_path).convert("RGB")
        prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
        messages = [{"role": "user", "content": prompt_content}]
        
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        output_tokens = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
        raw_response = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = raw_response.strip().replace("```json", "").replace("```", "")
        
        predictions = json.loads(json_str)
        return predictions, raw_response
    except Exception as e:
        raw_response += f"\n[PARSING ERROR]: {e}"
        return [], raw_response

# --- FUNCIONES DE VISUALIZACIÓN Y EVALUACIÓN ---

def draw_boxes(image, annotations, color, label_key):
    """Dibuja bounding boxes en una imagen."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for ann in annotations:
        box = ann.get('bounding_box')
        label = ann.get(label_key, 'Unknown')
        if not isinstance(box, list) or len(box) != 4: continue
        
        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height), (xmax * width, ymax * height)], outline=color, width=4)
        text_bbox = draw.textbbox((xmin * width, ymin * height), label, font=font)
        draw.rectangle((text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2), fill=color)
        draw.text((xmin * width, ymin * height), label, fill="black", font=font)
    return image

def create_comparison_image(image_path, predictions, ground_truths, output_path):
    """Crea una imagen de comparación lado a lado para la tarea de detección."""
    original_image = Image.open(image_path).convert("RGB")
    
    pred_img = draw_boxes(original_image.copy(), predictions, 'cyan', 'debris_type')
    gt_img = draw_boxes(original_image.copy(), ground_truths, 'lime', 'class_name')
    
    width, height = original_image.size
    title_height = 40
    comparison_img = Image.new('RGB', (width * 2, height + title_height), 'black')
    
    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text((10, 5), "Model Predictions (Cyan)", font=font, fill="cyan")
    draw.text((width + 10, 5), "Ground Truth (Green)", font=font, fill="lime")
    
    comparison_img.paste(pred_img, (0, title_height))
    comparison_img.paste(gt_img, (width, title_height))
    
    comparison_img.save(output_path)


def load_and_process_coco_annotations(coco_file_path):
    print(f"📖 Loading and processing COCO annotations from: {coco_file_path}")
    try:
        with open(coco_file_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Annotation file not found at {coco_file_path}")
        return None, None, None

    images_map = {img['id']: img for img in coco_data['images']}
    categories_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

    annotations_by_filename = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in images_map:
            image_info = images_map[image_id]
            filename = os.path.basename(image_info['file_name'])
            img_w, img_h = image_info['width'], image_info['height']
            
            coco_bbox = ann['bbox']
            x_min = coco_bbox[0] / img_w
            y_min = coco_bbox[1] / img_h
            x_max = (coco_bbox[0] + coco_bbox[2]) / img_w
            y_max = (coco_bbox[1] + coco_bbox[3]) / img_h

            annotations_by_filename[filename].append({
                "class_name": categories_map.get(ann['category_id'], "Unknown"),
                "bounding_box": [x_min, y_min, x_max, y_max]
            })
    
    return annotations_by_filename, list(annotations_by_filename.keys()), categories_map

def analyze_and_display_annotations_stats(ground_truths_map, categories_map):
    print("\n" + "="*50)
    print("📊 ANÁLISIS DEL DATASET DE VALIDACIÓN (GROUND-TRUTH)")
    print("="*50)
    num_images = len(ground_truths_map)
    all_annotations = [ann for anns in ground_truths_map.values() for ann in anns]
    num_annotations = len(all_annotations)
    num_classes = len(categories_map)

    print("\n--- Resumen General ---")
    print(f"  - Total de Imágenes en el set de validación: {num_images}")
    print(f"  - Total de Anotaciones (instancias): {num_annotations}")
    print(f"  - Número de Clases: {num_classes}")

    print("\n--- Distribución de Instancias por Clase ---")
    class_counts = defaultdict(int)
    for ann in all_annotations:
        class_counts[ann['class_name']] += 1

    print(f"{'Clase':<20} | {'Instancias':>10}")
    print("-" * 33)
    for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{class_name:<20} | {count:>10}")
    print("-" * 33)
    return class_counts


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    
    return intersection_area / union_area if union_area > 0 else 0


def print_metrics_report(stats, mode, file=None):
    def log_print(message):
        print(message)
        if file:
            file.write(message + '\n')

    log_print("\n" + "="*50)
    log_print(f"📊 FINAL {mode.upper()} EVALUATION REPORT")
    log_print("="*50)

    total_tp = sum(s["tp"] for s in stats.values())
    total_fp = sum(s["fp"] for s in stats.values())
    total_fn = sum(s["fn"] for s in stats.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    log_print("\n--- Overall Metrics ---")
    log_print(f"  Precision: {overall_precision:.2%}")
    log_print(f"  Recall:    {overall_recall:.2%}")
    log_print(f"  F1-Score:  {overall_f1:.2f}")

    log_print("\n--- Per-Class Metrics ---")
    log_print(f"{'Class':<20} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
    log_print("-" * 57)
    
    for class_name, s in sorted(stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        log_print(f"{class_name:<20} | {precision:>9.2%} | {recall:>9.2%} | {f1:>9.2f}")


def generate_visual_report(detection_stats, classification_stats, class_counts):
    """Genera y guarda un dashboard visual con todas las métricas de la evaluación."""
    
    def get_metrics(stats):
        metrics = {}
        for class_name, s in stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
        return metrics

    det_metrics = get_metrics(detection_stats)
    cls_metrics = get_metrics(classification_stats)
    
    all_classes = sorted(list(set(det_metrics.keys()) | set(cls_metrics.keys())))
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('OceanGuard AI - Evaluation Dashboard', fontsize=24, weight='bold')

    # Plotting function for metric bars
    def plot_bars(ax, title, metrics):
        labels = all_classes
        precision = [metrics.get(c, {}).get('precision', 0) for c in labels]
        recall = [metrics.get(c, {}).get('recall', 0) for c in labels]
        f1 = [metrics.get(c, {}).get('f1', 0) for c in labels]
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', color='royalblue')
        ax.bar(x, recall, width, label='Recall', color='limegreen')
        ax.bar(x + width, f1, width, label='F1-Score', color='tomato')
        
        ax.set_ylabel('Scores')
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plot_bars(axs[0, 0], 'Detection Performance (IoU-based)', det_metrics)
    plot_bars(axs[0, 1], 'Classification Performance (Presence-based)', cls_metrics)

    # Class Distribution Plot
    sorted_counts = sorted(class_counts.items(), key=lambda item: item[1])
    class_labels = [item[0] for item in sorted_counts]
    counts = [item[1] for item in sorted_counts]
    axs[1, 0].barh(class_labels, counts, color='skyblue')
    axs[1, 0].set_xlabel('Number of Instances')
    axs[1, 0].set_title('Ground-Truth Class Distribution', fontsize=16, weight='bold')
    axs[1, 0].grid(axis='x', linestyle='--', alpha=0.7)

    # Overall Metrics Table
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Overall Metrics Summary', fontsize=16, weight='bold')
    
    table_data = []
    for mode, stats in [('Detection', detection_stats), ('Classification', classification_stats)]:
        total_tp = sum(s["tp"] for s in stats.values())
        total_fp = sum(s["fp"] for s in stats.values())
        total_fn = sum(s["fn"] for s in stats.values())
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        table_data.append([mode, f"{precision:.2%}", f"{recall:.2%}", f"{f1:.2f}"])

    table = axs[1, 1].table(cellText=table_data,
                           colLabels=['Task', 'Precision', 'Recall', 'F1-Score'],
                           loc='center',
                           cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(REPORT_FILE)
    print(f"\n✅ Visual report saved to '{REPORT_FILE}'")


# --- SCRIPT PRINCIPAL ---

def main():
    parser = argparse.ArgumentParser(
        description="Run dual evaluation for OceanGuard AI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output for both tasks.")
    parser.add_argument('--draw', action='store_true', help="Generate visual comparisons for the detection task.")
    parser.add_argument('--report', action='store_true', help="Generate and save a visual report of the metrics.")
    args = parser.parse_args()

    model, tokenizer = setup_model()
    ground_truths_map, image_files_from_coco, categories_map = load_and_process_coco_annotations(ANNOTATION_FILE)
    
    if ground_truths_map is None: return

    print(f"\n🔍 Verificando imágenes existentes en el directorio: {IMAGE_DIR}")
    if not os.path.isdir(IMAGE_DIR):
        print(f"❌ ERROR: El directorio de imágenes no existe: {IMAGE_DIR}")
        return
        
    actual_files_in_dir = set(os.listdir(IMAGE_DIR))
    image_files_to_process = [f for f in image_files_from_coco if f in actual_files_in_dir]
    
    print(f"  - {len(image_files_from_coco)} imágenes encontradas en el fichero de anotaciones.")
    print(f"  - {len(actual_files_in_dir)} imágenes encontradas en el directorio.")
    print(f"  - {len(image_files_to_process)} imágenes en común se procesarán.")
    
    if not image_files_to_process:
        print("❌ No hay imágenes en común para procesar. Revisa las rutas.")
        return

    valid_class_names = list(categories_map.values())
    class_counts = analyze_and_display_annotations_stats(ground_truths_map, categories_map)

    if args.draw:
        os.makedirs(VISUALS_DIR, exist_ok=True)
        print(f"\n🎨 Visual comparison images will be saved to '{VISUALS_DIR}/'")

    detection_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    classification_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
        log_f.write("OceanGuard AI - Dual Evaluation Log\n" + "="*40 + "\n\n")

        print(f"\n🔬 Starting dual evaluation on {len(image_files_to_process)} images...")
        
        for i, filename in enumerate(image_files_to_process):
            print(f"  -> Processing image {i+1}/{len(image_files_to_process)}: {filename}")
            log_f.write(f"--- Image: {filename} ---\n")
            
            image_path = os.path.join(IMAGE_DIR, filename)

            detection_prompt = get_detection_prompt(valid_class_names)
            detection_preds, detection_raw = run_inference(model, tokenizer, image_path, detection_prompt)
            for det in detection_preds:
                if isinstance(det, dict) and 'debris_type' in det:
                    det['debris_type'] = det['debris_type'].strip().title()

            classification_prompt = get_classification_prompt(valid_class_names)
            classification_preds, classification_raw = run_inference(model, tokenizer, image_path, classification_prompt)

            if args.verbose:
                print("    --- Detection Raw Output ---")
                print("    " + detection_raw.replace("\n", "\n    "))
                print("    --- Classification Raw Output ---")
                print("    " + classification_raw.replace("\n", "\n    "))

            ground_truths = ground_truths_map.get(filename, [])
            gt_classes = set(ann['class_name'] for ann in ground_truths)
            
            log_f.write(f"\n-- DETECTION TASK --\n")
            log_f.write(f"MODEL RAW OUTPUT:\n{detection_raw}\n\nPARSED PREDICTIONS ({len(detection_preds)}):\n{json.dumps(detection_preds, indent=2)}\n\n")
            log_f.write(f"\n-- CLASSIFICATION TASK --\n")
            log_f.write(f"MODEL RAW OUTPUT:\n{classification_raw}\n\nPARSED PREDICTIONS ({len(classification_preds)}):\n{json.dumps(classification_preds, indent=2)}\n\n")
            log_f.write(f"\n-- GROUND TRUTH --\nAnnotations ({len(ground_truths)}):\n{json.dumps(ground_truths, indent=2)}\n\n")

            if args.draw:
                create_comparison_image(image_path, detection_preds, ground_truths, os.path.join(VISUALS_DIR, f"comp_{filename}"))

            gt_matched = [False] * len(ground_truths)
            for pred in detection_preds:
                pred_box, pred_class = pred.get("bounding_box"), pred.get("debris_type", "Unknown")
                if not isinstance(pred_box, list) or len(pred_box) != 4:
                    detection_stats[pred_class]["fp"] += 1; continue
                
                best_iou, best_gt_idx = 0, -1
                for j, gt in enumerate(ground_truths):
                    if gt["class_name"].lower() == pred_class.lower():
                        iou = calculate_iou(pred_box, gt["bounding_box"])
                        if iou > best_iou: best_iou, best_gt_idx = iou, j
                
                if best_iou >= IOU_THRESHOLD and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                    detection_stats[ground_truths[best_gt_idx]["class_name"]]["tp"] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    detection_stats[pred_class]["fp"] += 1
            
            for j, gt in enumerate(ground_truths):
                if not gt_matched[j]: detection_stats[gt["class_name"]]["fn"] += 1

            pred_classes = set(cls.strip().title() for cls in classification_preds if isinstance(cls, str))
            tp_classes = gt_classes.intersection(pred_classes)
            fp_classes = pred_classes.difference(gt_classes)
            fn_classes = gt_classes.difference(pred_classes)

            for cls in tp_classes: classification_stats[cls]["tp"] += 1
            for cls in fp_classes: classification_stats[cls]["fp"] += 1
            for cls in fn_classes: classification_stats[cls]["fn"] += 1
            
            if (i + 1) % LOG_INTERVAL == 0 or (i + 1) == len(image_files_to_process):
                print(f"   [Progreso {i+1}/{len(image_files_to_process)}]")
                for mode, stats in [('Detección', detection_stats), ('Clasificación', classification_stats)]:
                    tp = sum(s["tp"] for s in stats.values())
                    fp = sum(s["fp"] for s in stats.values())
                    fn = sum(s["fn"] for s in stats.values())
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    print(f"     - {mode}: Precisión: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1:.2f}")

        print_metrics_report(detection_stats, 'detection')
        print_metrics_report(classification_stats, 'classification')
        
        print_metrics_report(detection_stats, 'detection', file=log_f)
        print_metrics_report(classification_stats, 'classification', file=log_f)
    
    if args.report:
        print("\n🎨 Generando informe visual...")
        generate_visual_report(detection_stats, classification_stats, class_counts)

    print(f"\n✅ Evaluation finished. See '{LOG_FILE}' for detailed results.")

if __name__ == "__main__":
    main()