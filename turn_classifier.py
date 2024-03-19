import torch
import torch.nn as nn
from vision_encoders import VisionEncoder
import numpy as np

class TurnClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, enc_name="vit-base", num_frames=2):
        super(TurnClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_frames = num_frames

        self.encoder = VisionEncoder(enc_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), pretrained=True)
        self.dropout = nn.Dropout(p=0.)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.embed_dim * self.num_frames, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, num_frames, 3, 224, 224)
        # x = x.permute(1, 0, 2, 3, 4)
        embs = []
        for i in range(self.num_frames):
            print("image shape: ", x[i].shape)
            emb = self.encoder(x[i])
            embs.append(emb)
        embs = torch.stack(embs, dim=1)
        embs = embs.view(embs.shape[0], -1) # (batch_size, num_frames * embed_dim)
        x = self.mlp(embs) # (batch_size, num_classes)
        x = self.softmax(x) # (batch_size, num_classes)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurnClassifier(hidden_dim=1024, num_classes=2, enc_name="vit-base", num_frames=2).to(device)
    print(model)
    # x = torch.randn((2, 2, 3, 224, 224)) # (batch_size, num_frames, 3, 224, 224)
    x = np.random.randint(0, 256, (2, 2, 224, 224, 3))
    res = model(x)
    print(res)