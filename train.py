import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import wandb
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, hgg_dir, lgg_dir, transform=None):
        self.hgg_dir = hgg_dir
        self.lgg_dir = lgg_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        for img_name in os.listdir(self.hgg_dir):
            img_path = os.path.join(self.hgg_dir, img_name)
            self.images.append(img_path)
            self.labels.append(1)
        for img_name in os.listdir(self.lgg_dir):
            img_path = os.path.join(self.lgg_dir, img_name)
            self.images.append(img_path)
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CustomTestDataset(Dataset):
    def __init__(self, hgg_test_dir, lgg_test_dir, num_images_per_class=200, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        hgg_images = os.listdir(hgg_test_dir)[:num_images_per_class]
        for img in hgg_images:
            self.images.append(os.path.join(hgg_test_dir, img))
            self.labels.append(1)
        lgg_images = os.listdir(lgg_test_dir)[:num_images_per_class]
        for img in lgg_images:
            self.images.append(os.path.join(lgg_test_dir, img))
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def prepare_datasets(dataset_dir, dataset_ratio):
    hgg_dir = os.path.join(dataset_dir, 'training/HGG_training')
    lgg_dir = os.path.join(dataset_dir, 'training/LGG_training')
    hgg_test_dir = os.path.join(dataset_dir, 'testing/HGG_test')
    lgg_test_dir = os.path.join(dataset_dir, 'testing/LGG_test')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(hgg_dir, lgg_dir, transform=train_transform)
    test_dataset = CustomTestDataset(hgg_test_dir, lgg_test_dir, num_images_per_class=200, transform=test_transform)

    num_train_samples = int(len(train_dataset) * dataset_ratio)
    min_test_samples = 20
    num_test_samples = max(int(len(test_dataset) * dataset_ratio), min_test_samples)

    train_subset, _ = random_split(train_dataset, [num_train_samples, len(train_dataset) - num_train_samples])
    test_subset, _ = random_split(test_dataset, [num_test_samples, len(test_dataset) - num_test_samples])

    return train_subset, test_subset


# Función para obtener las activaciones de las capas usando hooks
def register_hooks(model):
    activations = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Buscar la última capa convolucional del modelo
    last_conv_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer  # Guarda la última capa convolucional

    if last_conv_layer is None:
        raise ValueError("No se encontró una capa convolucional en el modelo.")

    # Registrar el hook en la salida de la capa convolucional
    last_conv_layer.register_forward_hook(save_activation('last_conv'))

    return activations


def generate_grad_cam(model, input_image, pred_class, activations, device):
    # Asegurarse de que la imagen esté en el dispositivo correcto
    input_image = input_image[0].unsqueeze(0).to(device)  # Tomamos solo una imagen del batch

    # Desactivar gradientes para la imagen de entrada
    input_image.requires_grad = True

    # Realizar la predicción
    output = model(input_image)

    # Crear un tensor de gradientes del tamaño adecuado
    grad_output = torch.zeros_like(output)
    grad_output[0][pred_class] = 1  # Establecemos la clase predicha para el Grad-CAM

    # Retropropagar los gradientes
    model.zero_grad()
    output.backward(grad_output, retain_graph=True)

    # Obtener los gradientes sobre la imagen
    gradients = input_image.grad  # Gradientes con respecto a la imagen de entrada

    # Obtener las activaciones de la última capa convolucional
    feature_map = activations['last_conv']

    # **Redimensionar los gradientes**: Reducimos los gradientes a la misma escala de las activaciones
    gradients_resized = F.interpolate(gradients, size=(feature_map.shape[2], feature_map.shape[3]), mode='bilinear', align_corners=False)

    # **Promediar los gradientes**: Promediamos los gradientes sobre los 3 canales de la imagen.
    pooled_gradients = torch.mean(gradients_resized, dim=1, keepdim=True)  # Promediamos sobre los 3 canales de la imagen

    # Expandimos los gradientes para que coincidan con las activaciones
    pooled_gradients = pooled_gradients.expand_as(feature_map)  # Expandir a la forma de [1, 1280, 7, 7]

    # Multiplicar las activaciones por los gradientes ponderados
    weighted_activations = pooled_gradients * feature_map  # Multiplicamos cada canal por su gradiente

    # Sumar a través de los canales para obtener el mapa de activación
    grad_cam_map = torch.sum(weighted_activations, dim=1).squeeze()  # Sumamos a través de los canales

    # Asegurarnos de que el grad_cam_map sea 2D
    grad_cam_map = grad_cam_map.cpu().detach().numpy()
    grad_cam_map = np.maximum(grad_cam_map, 0)  # Eliminar valores negativos
    grad_cam_map = cv2.resize(grad_cam_map, (input_image.size(3), input_image.size(2)))  # Redimensionamos al tamaño de la imagen
    grad_cam_map -= np.min(grad_cam_map)
    grad_cam_map /= np.max(grad_cam_map)

    return grad_cam_map


def show_grad_cam(grad_cam_map, input_image, colormap=cv2.COLORMAP_JET):
    # Asegúrarse de que la imagen de entrada sea un tensor en el rango [0, 1]
    img = input_image[0].cpu().detach().numpy()  # Tomamos la primera imagen del batch

    # Redimensionar la imagen a [H, W, C] si está en [C, H, W]
    img = np.transpose(img, (1, 2, 0))  # Cambiar de [C, H, W] a [H, W, C]
    img = np.uint8(255 * img)  # Convertimos la imagen a formato uint8

    # Redimensionar el grad_cam_map al tamaño de la imagen original
    grad_cam_map_resized = cv2.resize(grad_cam_map, (img.shape[1], img.shape[0]))  # Redimensionar Grad-CAM

    # Normalizar el mapa de activación Grad-CAM entre 0 y 1
    grad_cam_map_resized = np.maximum(grad_cam_map_resized, 0)  # Eliminar valores negativos
    grad_cam_map_resized -= np.min(grad_cam_map_resized)  # Normalizamos el mapa de activación
    grad_cam_map_resized /= np.max(grad_cam_map_resized)  # Normalizamos el mapa de activación

    # Convertir el mapa de calor a una imagen de 3 canales (RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map_resized), colormap)

    # Asegurarse de que ambas imágenes tengan el mismo tamaño (alto, ancho)
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Superponer el Grad-CAM sobre la imagen original
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)  # Superponer Grad-CAM en la imagen original

    # Mostrar la imagen superpuesta
    plt.imshow(superimposed_img)
    plt.axis('off')  # Quitar los ejes
    plt.show()
    # Registrar el Grad-CAM en WandB
    wandb.log({"Test Grad-CAM": wandb.Image(superimposed_img)})





def train_and_evaluate(args):
    train_subset, test_subset = prepare_datasets(args.dataset_dir, args.dataset_ratio)

    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch, shuffle=False)

    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subset)):
        wandb.init(project=args.project_name, entity=args.team_name)

        train_fold_subset = Subset(train_subset, train_idx)
        val_fold_subset = Subset(train_subset, val_idx)

        train_loader = DataLoader(train_fold_subset, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_fold_subset, batch_size=args.batch, shuffle=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            all_labels = []
            all_outputs = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

            train_accuracy = accuracy_score(all_labels, all_outputs)
            train_precision = precision_score(all_labels, all_outputs, average='binary')
            train_recall = recall_score(all_labels, all_outputs, average='binary')
            train_f1 = f1_score(all_labels, all_outputs, average='binary')
            train_mcc = matthews_corrcoef(all_labels, all_outputs)  # Calcular MCC

            wandb.log({
                "Model": args.model,
                "Batch Size": args.batch,
                "Initial Learning Rate": args.lr,
                "Dataset Ratio": args.dataset_ratio,
                "Fold": fold + 1,
                "Epoch": epoch + 1,
                "Loss": running_loss / len(train_loader),
                "Accuracy": train_accuracy,
                "Precision": train_precision,
                "Recall": train_recall,
                "F1 Score": train_f1,
                "MCC": train_mcc,  # Añadir MCC a los logs
                "Final Learning Rate": optimizer.param_groups[0]['lr']
            })

            model.eval()
            val_loss = 0.0
            all_val_labels = []
            all_val_outputs = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
                    val_loss += loss.item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_outputs.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

            val_accuracy = accuracy_score(all_val_labels, all_val_outputs)
            val_precision = precision_score(all_val_labels, all_val_outputs, average='binary')
            val_recall = recall_score(all_val_labels, all_val_outputs, average='binary')
            val_f1 = f1_score(all_val_labels, all_val_outputs, average='binary')
            val_mcc = matthews_corrcoef(all_val_labels, all_val_outputs)  # Calcular MCC para la validación

            wandb.log({
                "Fold": fold + 1,
                "Validation Loss": val_loss / len(val_loader),
                "Validation Accuracy": val_accuracy,
                "Validation Precision": val_precision,
                "Validation Recall": val_recall,
                "Validation F1 Score": val_f1,
                "Validation MCC": val_mcc,  # Añadir MCC a los logs
            })

            scheduler.step(val_loss)

            model.eval()
            test_loss = 0.0
            all_test_labels = []
            all_test_outputs = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
                    test_loss += loss.item()
                    all_test_labels.extend(labels.cpu().numpy())
                    all_test_outputs.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())

            test_accuracy = accuracy_score(all_test_labels, all_test_outputs)
            test_precision = precision_score(all_test_labels, all_test_outputs, average='binary')
            test_recall = recall_score(all_test_labels, all_test_outputs, average='binary')
            test_f1 = f1_score(all_test_labels, all_test_outputs, average='binary')
            test_mcc = matthews_corrcoef(all_test_labels, all_test_outputs)  # Calcular MCC para el conjunto de prueba
            test_cm = confusion_matrix(all_test_labels, all_test_outputs)

            wandb.log({
                "Fold": fold + 1,
                "Test Loss": test_loss / len(test_loader),
                "Test Accuracy": test_accuracy,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1 Score": test_f1,
                "Test MCC": test_mcc,  # Añadir MCC al log
                
            })
              # **Grad-CAM después de la evaluación del fold completo**
            print(f"Generando Grad-CAM para una imagen del Fold {fold + 1}...")
            # Seleccionar una imagen de prueba para Grad-CAM (usaremos el DataLoader de test)
            #input_image, label = next(iter(test_loader))  # Obtener una imagen del DataLoader de test
            #input_image = input_image.to(device)
            # Tomar la primera imagen del batch para hacer la predicción
            #input_image = input_image[0].unsqueeze(0)  # Seleccionamos solo una imagen del batch
            # Extraer una imagen por clase contenida en test_loader
            class_images = {0: None, 1: None}  # Diccionario para almacenar una imagen por clase, in case multi class, change to N classes as required.
            for images, labels in test_loader:
                for img, lbl in zip(images, labels):
                    if class_images[lbl.item()] is None:
                        class_images[lbl.item()] = img.unsqueeze(0)  # Añadir dimensión de batch
                    if all(value is not None for value in class_images.values()):  # Si ya tenemos una imagen de cada clase, salir del bucle
                        break
                if all(value is not None for value in class_images.values()):  # Si ya tenemos una imagen de cada clase, salir del bucle
                    break

            # Registrar los hooks en el modelo
            activations = register_hooks(model)
            class_names = ['LGG', 'HGG']
            for class_idx, class_name in enumerate(class_names):
                print(f"Generando Grad-CAM para la clase: {class_name} (índice {class_idx})")
                grad_cam_map = generate_grad_cam(model, class_images[class_idx], pred_class=class_idx, activations=activations, device=device)
                # Mostrar el Grad-CAM
                show_grad_cam(grad_cam_map, class_images[class_idx], colormap=cv2.COLORMAP_JET)
            # Registrar los hooks en el modelo
            '''activations = register_hooks(model)
            class_names = ['LGG', 'HGG']
            for class_idx, class_name in enumerate(class_names):
                print(f"Generando Grad-CAM para la clase: {class_name} (índice {class_idx})")
                grad_cam_map = generate_grad_cam(model, input_image, pred_class=class_idx, activations=activations, device=device)
                # Mostrar el Grad-CAM
                show_grad_cam(grad_cam_map, input_image, colormap=cv2.COLORMAP_JET)
            '''
            plt.figure(figsize=(6, 6))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Low Grade', 'High Grade'], yticklabels=['Low Grade', 'High Grade'])
            plt.title(f'Test Confusion Matrix for Fold {fold + 1}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            wandb.log({"Test Confusion Matrix": wandb.Image(plt)})
            plt.tight_layout()
            plt.show()

        # Save model as .pt file
        model_path = f"model_fold_{fold + 1}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Log model to wandb as artifact
        artifact = wandb.Artifact(f"model_fold_{fold + 1}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        # Also save to wandb run directory
        wandb.save(model_path)

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on the dataset.')
    parser.add_argument('--project_name', type=str, required=True, help='WandB project name')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='Model name')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset_ratio', type=float, default=0.3, help='Ratio of the dataset to use for training and testing')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of k-folds for cross-validation')
    parser.add_argument('--team_name', type=str, default='computervision', help='WandB team name')

    args = parser.parse_args()
    train_and_evaluate(args)
