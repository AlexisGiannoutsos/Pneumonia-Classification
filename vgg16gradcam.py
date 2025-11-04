import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import os
import cv2


# Set device
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device}")

# Paths to dataset
train_dir = 'train'
val_dir = 'val'

# Hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 50
input_size = (224.224) #VGG16 input size

# Data augmentation (reduced aggressiveness)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# VGG16 base model
base_model = tf.keras.applications.VGG16(
    include_top=False, 
    weights='imagenet', 
    input_shape=(224, 224, 3),
    #name="vgg16"
)

# Unfreeze last 8 layers for better fine-tuning
for layer in base_model.layers[-8:]:
    layer.trainable = True

# Custom classifier with L2 regularization
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dropout(0.5),
    layers.Dense(
        train_generator.num_classes,
        activation='softmax',
        dtype='float32'
    )
])


# Class weights
class_counts = np.bincount(train_generator.classes)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'],run_eagerly=False)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
)

# Training
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stopping, lr_scheduler]
    #callbacks=[lr_scheduler]
)

# Save training history plot
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'.")

plot_history(history)

# Validation
def validate_model(model, val_generator):
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    class_names = list(train_generator.class_indices.keys())
    val_labels = val_generator.classes
    val_probs = model.predict(val_generator)
    val_preds = np.argmax(val_probs, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # Sensitivity and Specificity
    cm = confusion_matrix(val_labels, val_preds)
    print("\nSensitivity and Specificity per Class:")
    print(f"{'Class':<12}{'Sensitivity':>15}{'Specificity':>15}")
    print("-" * 42)
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f"{class_name:<12}{sensitivity:>15.2f}{specificity:>15.2f}")

    # AUC-ROC
    val_labels_bin = label_binarize(val_labels, classes=np.arange(len(class_names)))
    roc_auc = roc_auc_score(val_labels_bin, val_probs, average='macro', multi_class='ovr')
    print(f"\nAUC-ROC Score: {roc_auc:.4f}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'.")

    # ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc_vals = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(val_labels_bin[:, i], val_probs[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc_vals[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    print("ROC curves saved as 'roc_curves.png'.")

    # Inputs and outputs
    print(f"\nTotal number of validation inputs: {len(val_generator.filenames)}")
    print(f"Number of output classes: {len(class_names)}")

print("Validation started...")
validate_model(model, val_generator)

# Save model
model.save('pneumonia_vgg16tf.h5')
print("Model saved as 'pneumonia_vgg16tf.h5'")

#--------------------GRAD-CAM--------------

# --- Utility to confirm last conv layer ---
def show_VGG16_layers(model):
    print("\n--- VGG16 Backbone Layers ---")
    model.get_layer("vgg16").summary()
    print("--------------------------------\n")

# --- Auto-detect last conv layer if not provided ---
def infer_last_conv_layer_name(model, backbone_name="vgg16"):
    """
    Try to automatically find the last 4D conv/activation layer inside the backbone.
    Prints what it found so you can verify.
    """
    base = model.get_layer(backbone_name)

    # Check activation layers that still output 4D feature maps
    for layer in reversed(base.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            continue
        if hasattr(out_shape, "__len__") and len(out_shape) == 4:
            print(f"🔎 Using last 4D layer: {layer.name}, shape={out_shape}")
            return layer.name

    # Fallback if nothing found
    print("⚠️ Could not auto-detect last conv layer, falling back to 'block5_conv3'")
    return "block5_conv3"


# --- Fixed Grad-CAM function ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Grad-CAM for a Sequential model whose first layer is an Xception backbone.
    Uses backbone.input as the graph input and rebuilds the classifier head so
    the graph stays connected (avoids 'Graph disconnected').
    """
    # -------- Locate backbone & last conv layer --------
    backbone = model.get_layer("vgg16")
    if last_conv_layer_name is None:
        last_conv_layer_name = infer_last_conv_layer_name(model)  # your helper
    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    # -------- Rebuild the classifier head on top of backbone.output --------
    head_layers = []
    seen_backbone = False
    for lyr in model.layers:
        if seen_backbone:
            head_layers.append(lyr)
        if lyr is backbone or lyr.name == backbone.name:
            seen_backbone = True

    x = backbone.output
    for lyr in head_layers:
        # call layers with training=False to avoid Dropout randomness
        x = lyr(x, training=False)
    full_output = x  # same tensor as model.output, but connected to backbone.input

    # -------- Build Grad-CAM model (fully connected graph) --------
    grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, full_output]
    )

    # -------- Forward + gradients --------
    img_tensor = tf.cast(img_array, tf.float32)  # mixed precision safety
    with tf.GradientTape() as tape:
        conv_maps, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_maps)
    conv_maps = tf.cast(conv_maps, tf.float32)
    grads = tf.cast(grads, tf.float32)

    # -------- Grad-CAM computation (channel-wise weighted sum) --------
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_maps = conv_maps[0]                               # (H, W, C)
    heatmap = tf.reduce_sum(conv_maps * pooled_grads, axis=-1)  # (H, W)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        # fallback shape based on conv feature map size
        h, w = int(conv_maps.shape[0]), int(conv_maps.shape[1])
        return np.zeros((h, w), dtype=np.float32)

    heatmap = heatmap / max_val
    return heatmap.numpy()


# --- Overlay helper ---
def overlay_text_on_image(img_rgb, text, pos=(10,30), font_scale=0.8, thickness=2):
    cv2.putText(img_rgb, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img_rgb, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return img_rgb

# --- Save single Grad-CAM example ---
def save_and_display_gradcam(
    img_path,
    heatmap,
    filename="gradcam_example.png",
    alpha=0.4,
    output_dir=None,
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cam_path = os.path.join(output_dir, filename)
    else:
        cam_path = filename

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Grad-CAM saved to {cam_path}")
    return cam_path

# --- Save with labels ---
def save_gradcam_with_labels(
    img_path,
    heatmap,
    true_label,
    pred_label,
    confidence,
    filename,
    alpha=0.4,
    output_dir=None,
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
    else:
        out_path = filename

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1.0 - alpha, heatmap_color, alpha, 0)
    label_text = f"True: {true_label} | Pred: {pred_label} ({confidence:.2f})"
    overlay_text_on_image(superimposed_img, label_text, pos=(10, 30), font_scale=0.8, thickness=2)

    cv2.imwrite(out_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Grad-CAM with labels saved to {out_path}")
    return out_path

# --- Multi-example Grad-CAM generator ---
def generate_gradcam_examples(
    model,
    val_generator,
    class_names,
    last_conv_layer_name=None,
    examples_per_class=20,
    alpha=0.4,
    output_dir=None,
):
    saved_files = []
    captions_file = os.path.join(output_dir, "gradcam_captions.txt") if output_dir else "gradcam_captions.txt"
    with open(captions_file, "w") as cf:
        cf.write("Grad-CAM examples\n\n")

    class_counts = {c: 0 for c in class_names}
    TARGET_SIZE = (224, 224)

    for img_path, label in zip(val_generator.filepaths, val_generator.classes):
        true_class = class_names[label]
        if class_counts[true_class] >= examples_per_class:
            continue

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=TARGET_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        pred_class = class_names[pred_idx]
        confidence = float(preds[0][pred_idx])

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_idx)

        safe_true = true_class.replace(" ", "_")
        safe_pred = pred_class.replace(" ", "_")
        filename = f"{safe_true}_{class_counts[true_class]+1}_pred{safe_pred}_{confidence:.2f}.png"

        out_path = save_gradcam_with_labels(img_path, heatmap, true_class, pred_class, confidence, filename, alpha=alpha, output_dir=output_dir)
        saved_files.append(out_path)

        caption = (
            f"{os.path.basename(out_path)}: Grad-CAM overlay for a {true_class} image. "
            f"Model predicted '{pred_class}' with confidence {confidence:.2f}. "
            f"The highlighted regions indicate where the model focused when making its prediction."
        )
        with open(captions_file, "a") as cf:
            cf.write(caption + "\n\n")

        class_counts[true_class] += 1
        if all(count >= examples_per_class for count in class_counts.values()):
            break

    print(f"✅ Saved {len(saved_files)} Grad-CAM examples.")
    print(f"📝 Captions file: {captions_file}")
    print(f"🖼️ Files: {saved_files}")
    return saved_files, captions_file

# --- Run Grad-CAM automatically after training/validation ---
def run_gradcam_pipeline(model, val_generator, output_dir=None):
    try:
        class_names = list(val_generator.class_indices.keys())
        last_conv_layer_name = infer_last_conv_layer_name(model)
        print(f"\nRunning Grad-CAM generation using layer: {last_conv_layer_name} ...")
        saved_files, captions_file = generate_gradcam_examples(
            model, val_generator, class_names,
            last_conv_layer_name=last_conv_layer_name,
            examples_per_class=20,
            alpha=0.4,
            output_dir=output_dir
        )
        print("Grad-CAM pipeline complete.")
        print(f"Generated {len(saved_files)} heatmaps.")
    except Exception as e:
        print(f"[Grad-CAM] ❌ Failed: {e}")


# ✅ Call Grad-CAM AFTER defining everything
run_gradcam_pipeline(model, val_generator, output_dir=None)  # None => save in working directory
