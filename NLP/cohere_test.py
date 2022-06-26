# from tokens import COHERE_TOKEN, COHERE_MODEL_URL

import cohere
from cohere.classify import Example

co = cohere.Client(COHERE_TOKEN)

classifications = co.classify(
  model=COHERE_MODEL_URL,
  inputs=["I am so happy"]
)

print('The confidence levels of the labels are: {}'.format(
       classifications.classifications))