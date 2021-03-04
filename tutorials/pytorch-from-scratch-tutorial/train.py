# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from azureml.core import Run
from PIL import Image
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
import argparse
import io
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import torch
import torchvision
import torchsummary


# These are the classes in VOC
VOC_CLASSES = [         # Original Class Index
    "background",       # 0
    "airplane",         # 1
    "bicycle",          # 2
    "bird",             # 3
    "boat",             # 4
    "bottle",           # 5
    "bus",              # 6
    "car",              # 7
    "cat",              # 8
    "chair",            # 9
    "cow",              # 10
    "dining table",     # 11
    "dog",              # 12
    "horse",            # 13
    "motorbike",        # 14
    "person",           # 15
    "potted plant",     # 16
    "sheep",            # 17
    "sofa",             # 18
    "train",            # 19
    "tv/monitor"        # 20
]

# Combine a bunch of these so that we don't need a model with enough capacity to learn 20 classes.
# This will be more suitable for our embedded use case.
VOC_CLASSES_COMBINED = [
    "background",   # 0 Just maps to background
    "person",       # 1 Just maps to person
    "animal",       # 2 Will map to bird, cat, cow, dog, horse, sheep
    "vehicle",      # 3 Will map to airplane, bicycle, boat, bus, car, motorbike, train
    "indoor"        # 4 Will map to bottle, chair, dining table, potted plant, sofa, tv/monitor
                    #   I find that this class is quite noisily labeled. Best to ignore it most likely.
]

# We map all the original class indices into the small subset of classes we want to use.
VOC_SUBSET_MAPPING = {
    0: VOC_CLASSES_COMBINED.index("background"),
    1: VOC_CLASSES_COMBINED.index("vehicle"),
    2: VOC_CLASSES_COMBINED.index("vehicle"),
    3: VOC_CLASSES_COMBINED.index("animal"),
    4: VOC_CLASSES_COMBINED.index("vehicle"),
    5: VOC_CLASSES_COMBINED.index("indoor"),
    6: VOC_CLASSES_COMBINED.index("vehicle"),
    7: VOC_CLASSES_COMBINED.index("vehicle"),
    8: VOC_CLASSES_COMBINED.index("animal"),
    9: VOC_CLASSES_COMBINED.index("indoor"),
   10: VOC_CLASSES_COMBINED.index("animal"),
   11: VOC_CLASSES_COMBINED.index("indoor"),
   12: VOC_CLASSES_COMBINED.index("animal"),
   13: VOC_CLASSES_COMBINED.index("animal"),
   14: VOC_CLASSES_COMBINED.index("vehicle"),
   15: VOC_CLASSES_COMBINED.index("person"),
   16: VOC_CLASSES_COMBINED.index("indoor"),
   17: VOC_CLASSES_COMBINED.index("animal"),
   18: VOC_CLASSES_COMBINED.index("indoor"),
   19: VOC_CLASSES_COMBINED.index("vehicle"),
   20: VOC_CLASSES_COMBINED.index("indoor")
}

# Here are the RGB values for each class
VOC_SUBSET_COLORS = [
    [  0,   0,   0],  # 0 -> background -> black
    [128,   0,   0],  # 1 -> person -> red
    [  0, 128,   0],  # 2 -> animal -> green
    [128, 128,   0],  # 3 -> vehicle -> yellow
    [  0,   0, 128],  # 4 -> indoor -> blue
]

class ConfusionMatrix:
    """
    A confusion matrix for semantic segmentation that can be interpreted by
    both Matplotlib and AML logging.
    """
    def __init__(self, class_names):
        n = len(class_names)
        self._class_names = class_names
        self._array = np.zeros((n, n), dtype=int)
        # Array is row=true, col=pred

    def update(self, predmask, gtmask):
        """
        Updates the confusion matrix with the given information.

        Calculates how many pixels were classified correctly vs misclassified,
        treating each pixel as a separate prediction towards each class.

        Each mask should be a PyTorch Tensor object of shape {H, W}.
        """
        # For each pixel, load it into the array at array[row=gt class idx][col=pred class idx]
        assert predmask.shape == gtmask.shape, f"Shapes of predicted and gt do not match: {predmask.shape} and {gtmask.shape}"

        n = len(self._class_names)
        for r in range(n):
            for c in range(n):
                self._array[r][c] += np.sum((gtmask.cpu().detach().numpy() == r) & (predmask.cpu().detach().numpy() == c))

    def as_ndarray(self):
        """
        Returns ourselves as an ND Array, compatible with SKLearn's style of
        confusion matrix.
        """
        return self._array

class TransformedVocDataset(torchvision.datasets.VOCSegmentation):
    def __init__(self, dataset_dir, image_set, download, x_transforms, y_transforms):
        """
        Possibly downloads the given split of the VOC Segmentation dataset,
        and sets up the transforms for it.

        This class is needed because transforms that rely on random behavior
        need to be applied identically to both X and Y, but in the base class'
        case, this is not possible.

        Corresponding transforms must be aligned in x_transforms and y_transforms by index.
        For example, if you want to do something to X but not to Y, you can do that,
        but for any transform in X, say x_transforms[i], that you want
        to do to Y as well, make sure that y_transforms[i] is the same function.
        """
        super().__init__(dataset_dir, image_set=image_set, download=download)
        self._x_transforms = x_transforms if x_transforms else []
        self._y_transforms = y_transforms if y_transforms else []
        self._palette = super().__getitem__(0)[1].getpalette() if len(self) > 0 else None

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Now set the random seed to be identical across all the possible sources of random behavior
        seed = np.random.randint(8675309)
        random.seed(seed)
        torch.manual_seed(seed)
        for transform in self._x_transforms:
            img = transform(img)

        # Set the random seed back to the original value so that any random generators will behave
        # identically going through the Y transforms as they did in the X transforms
        random.seed(seed)
        torch.manual_seed(seed)
        for transform in self._y_transforms:
            target = transform(target)

        # Map the target's values into the subset of values
        # This is a little funky. I am looping over all the class indices
        # and replacing them with N_CLASSES + j, where j is their new class index.
        # Then I am subtracting N_CLASSES from all the non-255 values.
        # This is so that I don't map class 1 to class 3 (say), and then when I get to
        # class 3, remap it to some other class. This bug took like an hour to figure out.
        target = np.array(target)
        for i in range(len(VOC_CLASSES)):
            target[target == i] = len(VOC_SUBSET_MAPPING) + VOC_SUBSET_MAPPING[i]
        target[(target >= len(VOC_SUBSET_MAPPING)) & (target != 255)] -= len(VOC_SUBSET_MAPPING)

        # Now convert the resulting mask image to a tensor
        target = torch.from_numpy(target)

        return img, target

    @staticmethod
    def collapse_one_hot_tensor(pred):
        """
        Collapses a one-hot tensor of shape {C, H, W} into a shape {H, W} tensor, where
        each value is the index of the maximum value of the softmax.
        """
        pred = pred.clone().detach()
        pred = torch.log_softmax(pred, dim=0)
        pred = torch.argmax(pred, dim=0)
        return pred

    def mask_tensor_to_pil_image(self, y):
        """
        Convert the given mask to a displayable PIL Image.
        """
        y = Image.fromarray(y.cpu().numpy(), mode="P")
        y.putpalette(self._palette)
        return y

    @staticmethod
    def one_hot_tensor_to_pil_image(pred):
        """
        Convert the given network output (of shape {C, H, W}) to displayable PIL Image.
        """
        img = TransformedVocDataset.collapse_one_hot_tensor(pred).cpu()
        img = img.unsqueeze(0).repeat(3, 1, 1).type(torch.uint8)
        for i in range(len(VOC_SUBSET_COLORS)):
            img[1][img[0] == i] = torch.tensor(VOC_SUBSET_COLORS[i], dtype=torch.uint8)[1]
            img[2][img[0] == i] = torch.tensor(VOC_SUBSET_COLORS[i], dtype=torch.uint8)[2]
            img[0][img[0] == i] = torch.tensor(VOC_SUBSET_COLORS[i], dtype=torch.uint8)[0]

        img = img.permute(1, 2, 0)
        img = Image.fromarray(img.numpy())
        return img

class UNet(torch.nn.Module):
    """
    This is the network we will be training.

    It is originally based on https://www.kaggle.com/cordmaur/38-cloud-simple-unet, and is taken under Apache 2.0.
    """
    def __init__(self, nclasses):
        super().__init__()

        rgb = 3
        self.conv1 = self.contract_block(rgb, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.conv4 = self.contract_block(128, 256, 3, 1)
        self.conv5 = self.contract_block(256, 512, 3, 1)

        self.upconv5 = self.expand_block(512, 256, 3, 1)
        self.upconv4 = self.expand_block(256*2, 128, 3, 1)
        self.upconv3 = self.expand_block(128*2, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, nclasses, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Upsampling part (including skip connections)
        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                                     torch.nn.BatchNorm2d(out_channels),
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                                     torch.nn.BatchNorm2d(out_channels),
                                     torch.nn.ReLU(),
                                     torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, writer, epoch, use_cuda=True, log=True):
    # Set to training mode
    model.train(True)

    print("Training")
    for x, y in tqdm(dataloader):
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        # Convert to Long
        y = y.long()

        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backprop through the network to update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log
        if log:
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Learning-Rate", scheduler.get_last_lr()[0])  # get_last_lr() returns a list

            # Log with AML
            run = Run.get_context()
            logloss = np.asscalar(torch.clone(loss).detach().cpu().numpy())
            run.log(name="Loss/train", value=logloss)
            run.log(name="Learning-Rate", value=scheduler.get_last_lr()[0])

def compute_classification_accuracy(pred, y):
    """
    Compute the classification accuracy on this one prediction. Computes the total number of pixels that
    should have been classified as each class, then computes the total number of pixels that were classified
    as each class (all irrespective of the locations of these pixels).
    Then computes the proportion for each class (by placing the smaller number on top in each class), and sums the result.

    This should range from 0 (in the case that no pixels were assigned to classes that are actually present)
    to 1 (in the case that the right number of pixels were assigned in each class, regardless of location).
    """
    ntotal_pixels = torch.numel(pred)

    classacc = 0.0
    for classidx in range(len(VOC_CLASSES_COMBINED)):
        n_should_have_this_class = (y == classidx).sum().item()
        n_was_this_class = (TransformedVocDataset.collapse_one_hot_tensor(pred.squeeze()) == classidx).sum().item()

        if n_should_have_this_class == 0 and n_was_this_class == 0:
            classacc += 1.0
            continue

        if n_should_have_this_class <= n_was_this_class:
            classacc += n_should_have_this_class / n_was_this_class
        else:
            classacc += n_was_this_class / n_should_have_this_class

    return classacc / len(VOC_CLASSES_COMBINED)

def compute_loc_acc(pred, y):
    """
    Compute the localization accuracy on this one prediction. I.e., computes the number of pixels that were
    correctly assigned to background plus the total number of pixels that were correctly assigned NOT
    background (regardless of which class they were actually assigned to) divided by the total number
    of pixels.
    """
    pred = TransformedVocDataset.collapse_one_hot_tensor(pred.squeeze())
    true_background = (y == VOC_CLASSES_COMBINED.index("background"))
    guessed_background = (pred == VOC_CLASSES_COMBINED.index("background"))
    correctly_guessed_background = torch.sum((guessed_background & true_background)).item()

    true_not_background = (y != VOC_CLASSES_COMBINED.index("background"))
    guessed_not_background = (pred != VOC_CLASSES_COMBINED.index("background"))
    correctly_guessed_not_background = torch.sum((guessed_not_background & true_not_background)).item()

    return (correctly_guessed_background + correctly_guessed_not_background) / torch.numel(pred)

def compute_dice(pred, y, weights=None):
    """
    Computes the multiclass dice coefficient of this one prediction.

    This is one of apparantly many ways to compute this metric... which maybe means
    it is not a great metric...

    Anyway, we just go through each class and compute the fraction of pixels that
    were located and classified for that class correctly, then multiply it by its weight.
    Then we sum up these values.

    The resulting value should be between 0 (meaning no pixels were correctly classified
    and located across all classes) and 1 (meaning all pixels were correctly classified
    and located across all classes).
    """
    pred = TransformedVocDataset.collapse_one_hot_tensor(pred.squeeze())

    if weights is None:
        weights = [1.0 for _ in range(len(VOC_CLASSES_COMBINED))]

    dice = 0.0
    for classidx in range(len(VOC_CLASSES_COMBINED)):
        if (y == classidx).sum().item() > 0.0:
            dice += weights[classidx] * ((pred == classidx) & (y == classidx)).sum().item() / (y == classidx).sum().item()
        else:
            dice += weights[classidx] * 1.0

    return dice / len(VOC_CLASSES_COMBINED)

def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Matplotlib. Stolen largely from Tensorflow examples
    under Apache 2.0 license.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot to PIL image and returns it.
    The supplied figure is closed and inaccessible after this call.

    This method taken mostly from Tensorflow tutorials, under Apache 2.0 license.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return image

def validate_one_epoch(model, dataloader, loss_fn, optimizer, writer, epoch, dataset, weights, use_cuda=True, log=True):
    # Turn off training mode
    model.train(False)

    losses = []
    accuracies = []
    localizations = []
    dices = []
    confmat = ConfusionMatrix(class_names=VOC_CLASSES_COMBINED)
    print("Validating")
    for x, y in tqdm(dataloader):
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y.long())
            losses.append(loss.cpu().item())
            accuracies.append(compute_classification_accuracy(pred, y))
            localizations.append(compute_loc_acc(pred, y))
            dices.append(compute_dice(pred, y, weights=weights))
            confmat.update(TransformedVocDataset.collapse_one_hot_tensor(pred.squeeze()), y.squeeze())

    # Log
    if log:
        # Log the loss value
        run = Run.get_context()
        writer.add_scalar("Loss/val", np.asscalar(np.mean(np.array(losses))), epoch)
        run.log(name="Loss/val", value=np.asscalar(np.mean(np.array(losses))))

        # Compose an image grid and log it for visualization
        imgs_for_grid = [
            x[0].cpu(),  # Tensor
            T.ToTensor()(dataset.mask_tensor_to_pil_image(y[0]).convert("RGB")).cpu(),        # Tensor -> PIL (with correct RGB values in palette) -> Tensor
            T.ToTensor()(TransformedVocDataset.one_hot_tensor_to_pil_image(pred[0]).convert("RGB")).cpu()   # Tensor -> PIL (with correct RGB values in palette) -> Tensor
        ]
        imggrid = torchvision.utils.make_grid(imgs_for_grid)
        writer.add_image("Images/val", imggrid, epoch)

        # Compose it into Matplotlib for AML plotting
        plt.title("Images/val")
        plt.imshow(imggrid.cpu().permute(1, 2, 0).numpy())
        run.log_image(name=f"Images/val_{epoch:04d}", plot=plt)

        # Log the classification metrics
        accuracy = np.asscalar(np.mean(np.array(accuracies)))
        writer.add_scalar("Accuracy/val", accuracy, epoch)
        run.log(name="Accuracy/val", value=accuracy)

        # Log the localization metrics
        localization = np.asscalar(np.mean(np.array(localizations)))
        writer.add_scalar("Localization/val", localization, epoch)
        run.log(name="Localization/val", value=localization)

        # Log dice metric
        dice_value = np.asscalar(np.mean(np.array(dices)))
        writer.add_scalar("Dice/val", dice_value, epoch)
        run.log(name="Dice/val", value=dice_value)

        # Log Confusion Matrix
        figure = plot_confusion_matrix(confmat.as_ndarray(), class_names=VOC_CLASSES_COMBINED)
        run.log_image(name="ConfusionMatrix/val", plot=figure)
        confmat_image = plot_to_image(figure)
        writer.add_image("ConfusionMatrix/val", T.ToTensor()(confmat_image), epoch)

def train(model, train_dloader, validation_dloader, loss_fn, optimizer, scheduler, dataset, weights, epochs=1, use_cuda=True, log=True):
    if use_cuda:
        model.cuda()

    # Set up TensorBoard for logging and visualization
    writer = SummaryWriter(log_dir="logs")

    for epoch in range(epochs):
        print("Epoch", epoch + 1, "of", epochs)
        train_one_epoch(model, train_dloader, loss_fn, optimizer, scheduler, writer, epoch, use_cuda=use_cuda, log=log)
        validate_one_epoch(model, validation_dloader, loss_fn, optimizer, writer, epoch, dataset, weights, use_cuda=use_cuda, log=log)

def get_transforms(size=128):
    """
    Gets all the torchvision transforms we will be applying to the dataset.
    """
    # These are the transformations that we will do to our dataset
    # For X transforms, let's do some of the usual suspects and convert to tensor.
    # Don't forget to normalize to [0.0, 1.0], FP32
    # and don't forget to resize to the same size every time.
    x_transforms = [
        T.Resize((size, size)),
        T.RandomApply([
            T.RandomAffine(degrees=20, translate=(0.1, 0.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-30, 30)),
            T.RandomVerticalFlip(p=0.5),
        ], p=0.5),
        T.ColorJitter(brightness=0.5),
        T.ToTensor(),  # Converts to FP32 [0.0, 1.0], Tensor type
    ]

    # For Y transforms, we need to make sure that we do the same thing to the ground truth,
    # since we are trying to recreate the image.
    y_transforms = [
        T.Resize((size, size), interpolation=Image.NEAREST),  # Make sure we don't corrupt the labels
        T.RandomApply([
            T.RandomAffine(degrees=20, translate=(0.1, 0.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-30, 30)),
            T.RandomVerticalFlip(p=0.5),
        ], p=0.5),
    ]

    return x_transforms, y_transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batchsize", "-b", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--dataset", "-d", type=str, default="dataset", help="Path to the root of the VOC dataset. If this folder does not exist, we download it, which takes about an hour.")
    parser.add_argument("--dataset-val", "-v", type=str, default="dataset-val", help="Path to the root of the VOC validation split.")
    parser.add_argument("--learning-rate", "-r", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--nepochs", "-e", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--split", "-s", type=float, default=0.7, help="Fraction in (0.0, 1.0) of the dataset to use for training. The rest will be reserved for validation.")
    parser.add_argument("--use-cpu", "-c", action="store_true", help="If given, we use CPU, otherwise we try to use CUDA.")
    parser.add_argument("--outputs", "-o", type=str, default="outputs", help="Path to save everything.")
    parser.add_argument("--weights", "-w", type=str, default=None, help="If given, should be a list of weights equal in length to the number of classes. We weight the loss with these values.")
    parser.add_argument("--no-logs", action="store_true", help="If given, we do not log.")
    parser.add_argument("--resize", type=int, default=128, help="Size of the images. We will resize to this value and use it as both H and W.")
    args = parser.parse_args()

    # Sanity check args
    if args.split <= 0.0 or args.split >= 1.0:
        print(f"Split is {args.split}, but must be in interval 0.0 < split < 1.0")
        exit(1)

    if args.weights is not None:
        args.weights = torch.tensor([float(val) for val in args.weights.split()])
        for w in args.weights:
            if w < 0.0:
                print("Weights should all be positive or zero, but got", w)
                exit(2)

    # We need to remove all the old TensorBoard logs and old models
    shutil.rmtree(args.outputs)
    for fname in os.listdir("logs"):
        if fname.startswith("events.out.tfevents"):
            os.remove(os.path.join("logs", fname))

    # This is where the dataset will go in our local workspace
    dataset_dir = args.dataset
    dataset_dir_val = args.dataset_val
    download = not os.path.isdir(dataset_dir)  # Let's not download if we've already downloaded before.
    download_val = not os.path.isdir(dataset_dir_val)

    # Now let's make the dataset
    x_transforms, y_transforms = get_transforms()
    dataset_train = TransformedVocDataset(dataset_dir, image_set="train", download=download, x_transforms=x_transforms, y_transforms=y_transforms)
    dataset_val = TransformedVocDataset(dataset_dir_val, image_set="val", download=download_val, x_transforms=x_transforms, y_transforms=y_transforms)
    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

    # How many images do we have?
    ntotal_imgs_in_dataset = len(dataset)

    # Let's split the dataset into train/val splits.
    ntrain_split = int(round(args.split * ntotal_imgs_in_dataset))
    nval_split = ntotal_imgs_in_dataset - ntrain_split
    print("N Train Split Images:", ntrain_split)
    print("N Validation Split Images:", nval_split)
    training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [ntrain_split, nval_split])

    # And create a dataloader to encapsulate each.
    train_split = DataLoader(training_dataset, batch_size=args.batchsize, shuffle=True)
    val_split = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    # Should we use CUDA?
    use_cuda = not args.use_cpu

    # Create the model itself
    unet = UNet(len(VOC_CLASSES_COMBINED))

    # Print Keras-style summarization
    if use_cuda:
        torchsummary.summary(unet.cuda(), (3, args.resize, args.resize))
    else:
        torchsummary.summary(unet, (3, args.resize, args.resize))

    # Set up the scheduler
    n_learning_rate_cycles = 4  # Cycle the learning rate 4 times: (min -> max, max -> min) x 4
    steps_per_epoch = int(math.ceil(ntrain_split / args.batchsize))
    total_steps = int(steps_per_epoch * args.nepochs)
    cycle_half_period = int(math.ceil(total_steps / (2 * n_learning_rate_cycles)))
    print(f"Will cycle the learning rate up for {cycle_half_period} batches, then down for that many, {n_learning_rate_cycles} times in total.")
    opt = torch.optim.SGD(unet.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.learning_rate, max_lr=0.1, step_size_up=cycle_half_period, mode="triangular2")

    # Training
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255, weight=args.weights)
    if use_cuda:
        loss_fn.cuda()
    train(unet, train_split, val_split, loss_fn, opt, scheduler, dataset_train, args.weights, epochs=args.nepochs, use_cuda=use_cuda, log=not args.no_logs)

    # Save
    os.makedirs(args.outputs, exist_ok=True)
    modelpath = os.path.join(args.outputs, "model.pth")
    torch.save(unet.state_dict(), modelpath)
    print("Model saved to", modelpath)