import os

from cv2.cv2 import resize, INTER_CUBIC

os.chdir('../style_transfer')
import torch
from PIL import Image
from app.libs.Matrix import MulLayer
from app.libs.models import encoder3
from app.libs.models import decoder3
import torchvision.transforms as transforms

class StyleTransferModel:
    def __init__(self, model_dir, fine_size=256):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on ", self.device)
        self.layer = 'r31'

        self.matrix = MulLayer(layer=self.layer)
        self.vgg = encoder3()
        self.dec = decoder3()

        self.load_model(model_dir)
        self.eval()
        if self.device == 'cuda':
            self.matrix.cuda()
            self.vgg.cuda()
            self.dec.cuda()
        self.fine_size = fine_size
        self.content = torch.Tensor(1, 3, self.fine_size, self.fine_size).to(self.device)

    def load_model(self, model_dir):
        self.vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg_r31.pth')))
        self.dec.load_state_dict(torch.load(os.path.join(model_dir, 'dec_r31.pth')))
        self.matrix.load_state_dict(torch.load(os.path.join(model_dir, 'r31.pth')))

    def eval(self):
        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.dec.parameters():
            param.requires_grad = False
        for param in self.matrix.parameters():
            param.requires_grad = False

    def load_style(self, imgPath):
        img = Image.open(imgPath).convert('RGB')
        transform = transforms.Compose([
                    transforms.Scale(self.fineSize),
                    transforms.ToTensor()])
        self.style = transform(img).unsqueeze(0).to(self.device)

    def inference(self, cap):
        result_frames = []

        with torch.no_grad():
            # sometimes doesn't work from first time lol
            try:
                sF = self.vgg(self.style)
            except:
                sF = self.vgg(self.style)

        while True:
            
            ret,frame = cap.read()
            if not ret:
                break
            frame = resize(frame,(512,256),interpolation=INTER_CUBIC)
            frame = frame.transpose((2,0,1))
            frame = frame[::-1,:,:]
            frame = frame/255.0

            frame = torch.from_numpy(frame.copy()).unsqueeze(0)
            self.content.data.resize_(frame.size()).copy_(frame)
            with torch.no_grad():
                cF = self.vgg(self.content)
                if(self.layer == 'r41'):
                    feature,transmatrix = self.matrix(cF[self.layer],sF[self.layer])
                else:
                    feature,transmatrix = self.matrix(cF,sF)
                transfer = self.dec(feature)
            transfer = transfer.clamp(0,1).squeeze(0).data.cpu().numpy()
            transfer = transfer.transpose((1,2,0))
            transfer = transfer[...,::-1]
            result_frames.append(transfer * 255)

        cap.release()
        return result_frames

