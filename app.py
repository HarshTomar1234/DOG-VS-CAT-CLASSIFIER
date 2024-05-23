
from fastai.vision.all import *
import gradio as gr


def is_cat(x):
  return x[0].isupper()


img = PILImage.create('dog.jpg')
img.thumbnail((192, 192))
img



learn =  load_learner('model.pkl')

learn.predict(img)


categories = ('Dog', 'Cat')

def classify_image(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float, probs)))

classify_image(img)


image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['dog.jpg', 'cat.jpg', 'rabbit.jpg']

intf = gr.Interface(fn= classify_image, inputs = image, outputs = label, examples = examples)
intf.launch(inline = False, share = True)