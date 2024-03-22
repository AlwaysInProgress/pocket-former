import torch
import torch.nn as nn
import timm
# import clip
import argparse
# from voltron import instantiate_extractor, load
# from vip import load_vip
import torchvision.transforms as T
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
# from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
# from segment_anything import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h

class VisionEncoder(nn.Module):
    def __init__(self, model_name, device, pretrained=True):
        super(VisionEncoder, self).__init__()
        self.device = device
        self.preprocess = None
        self.model_name = model_name

        # ViT
        if model_name == "vit-tiny":
            self.model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 192
        elif model_name == "vit-small":
            self.model = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 384
        elif model_name == "vit-base":
            self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 768
        elif model_name == "vit-large":
            self.model = timm.create_model("vit_large_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 1024
        
        # ResNet
        elif model_name == "resnet18":
            self.model = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
            self.embed_dim = 512
        elif model_name == "resnet50":
            self.model = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            self.embed_dim = 2048
        elif model_name == "resnet101":
            self.model = timm.create_model("resnet101", pretrained=pretrained, num_classes=0)
            self.embed_dim = 2048
        elif model_name == "resnet152":
            self.model = timm.create_model("resnet152", pretrained=pretrained, num_classes=0)
            self.embed_dim = 2048

        # CLIP
        elif model_name == "clip-base":
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            self.embed_dim = 768
        elif model_name == "clip-large":
            self.model, self.preprocess = clip.load("ViT-L/14", device=device)
            self.embed_dim = 1024
        elif model_name == "clip-resnet50":
            self.model, self.preprocess= clip.load("RN50", device=device)
            self.embed_dim = 512

        # DINOv2
        elif model_name == "dinov2-small":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.embed_dim = 384
        elif model_name == "dinov2-base":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits24')
            self.embed_dim = 768
        elif model_name == "dinov2-large":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits38')
            self.embed_dim = 1024

        # Voltron
        elif model_name == "voltron-base":
            self.model, self.preprocess = load("v-cond-base", device=device, freeze=True, cache="../voltron-robotics/cache")
            self.embed_dim = 768
            self.vector_extractor = instantiate_extractor(self.model)().to(device)

        # VIP
        elif model_name == "vip":
            self.model = load_vip()
            self.embed_dim = 2048

        # SAM
        elif model_name == "sam-base":
            sam_path = "/mnt/extdisk/sam_checkpoints/sam_vit_b_01ec64.pth"
            self.model = build_sam_vit_b(checkpoint=sam_path)
            self.predictor = SamPredictor(self.model)
            # self.preprocess = self.model.preprocess
            # self.preprocess = self.predictor.set_image
            self.embed_dim = 768

        else:
            raise ValueError(f"Model name {model_name} not recognized.")


        if self.preprocess is None:
            # self.preprocess = self.preprocess_numpy
            self.preprocess = lambda x: x

        self.model.to(device)
        self.model.eval()

    def preprocess_numpy(self, images):
        '''
        Preprocesses numpy images from cv2.imread for the model
        '''
        images = images.astype(np.float32)
        images = cv2.resize(images, (224, 224))
        images = torch.from_numpy(images).permute(2, 0, 1).float() / 255.0

        return images
    
    def preprocess_batch(self, images):
        '''
        Preprocesses a batch of images for the model
        '''
        processed_images = []
        for image in images:
            image = self.preprocess(image)
            processed_images.append(image)

        return torch.stack(processed_images)

    def forward(self, images):
        if images.ndim == 3:
            images = images[None, ...]

        if "voltron" in self.model_name:
            # convert to tensor using torchvision
            # images = self.preprocess(torch.tensor(images).permute(2, 0, 1).to(self.device))
            images = self.preprocess(torch.tensor(images).permute(0, 3, 1, 2).to(self.device))
        elif "clip" in self.model_name:
            # turning into PIL image
            images = [PIL.Image.fromarray(image) for image in images]
            images = self.preprocess_batch(images).to(self.device)
        elif "sam" in self.model_name:
            embs = []
            for image in images:
                print("image shape in sam forward: ", image.shape)
                self.predictor.set_image(image)
                embs.append(self.predictor.get_image_embedding())
            return torch.stack(embs).squeeze()
        else:
            images = self.preprocess_batch(images).to(self.device)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # display image for debugging
        # image = images[0].permute(1, 2, 0).detach().cpu().numpy()
        # print("image shape: ", image.shape)
        # plt.imshow(image)
        # plt.show()

        if "clip" in self.model_name:
            return self.model.encode_image(images)
        elif "voltron" in self.model_name:
            dense_emb =  self.model(images, mode="visual")
            return self.vector_extractor(dense_emb)
        else:
            return self.model(images)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit-base", help="Name of the model to use.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = VisionEncoder(args.model_name, device)
    print("model: ", encoder.model)
    print("preprocess: ", encoder.preprocess)

    # pil image
    # image = PIL.Image.open("/nethome/jcollins90/Downloads/mandelbrot.jpg")
    image = cv2.imread("/nethome/jcollins90/Downloads/mandelbrot.jpg")

    # simulating batch
    image = np.repeat(image[None, ...], 4, axis=0)

    print("image shape: ", image.shape)

    emb = encoder(image)
    print("embedding shape: ", emb.shape)