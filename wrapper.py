import os

import gradio as gr
import torch
import torch.nn
import torch.nn
import torchvision.transforms as transforms
from transformers import pipeline

from networks.resnet import resnet50


# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-f','--file', default='examples_realfakedir')
# parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
# parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
# parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
#
# opt = parser.parse_args()
def judge(img):
    model_path = 'weights/blur_jpg_prob0.5.pth'
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    trans_init = []
    trans = transforms.Compose(trans_init + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = trans(img.convert('RGB'))
    with torch.no_grad():
        in_tens = img.unsqueeze(0)
        if torch.cuda.is_available():
            in_tens = in_tens.cuda()
        prob = model(in_tens).sigmoid().item()
        return prob
    # print('probability of being synthetic: {:.2f}%'.format(prob * 100))


def classify(image):
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    pipe = pipeline("image-classification", model="./weights/sdxl-detector")
    # pipe2 = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    outputs = pipe(image)
    results = {}
    for result in outputs:
        results[result['label']] = result['score']
    return results


def run(img, name):
    print(img)
    if name.startswith('CNN'):
        r = judge(img)
        print(r)
        return {'artificial': r, 'human': 1 - r}
    else:
        return classify(img)


def web():
    # 打印结果
    title = "AI合成图像检测"
    description = "AI合成图像检测"
    # 输入包含模型选择和图像上传
    inputs = [gr.Image(type="pil", label="上传图片"),
              gr.Radio(["Organika/sdxl-detector", 'CNNDetection'], label="选择模型")]
    outputs = [gr.Label(num_top_classes=2)]
    demo = gr.Interface(fn=run, inputs=inputs, outputs=outputs, title=title, description=description,
                        allow_flagging='never')
    demo.launch(show_api=False)
if __name__ == '__main__':
    web()