## word2vec-tf-model
word2vec model by tf.contrib.lookup.HashTable & tf.nn.embedding_lookup

I use tf 1.9.0 for operation check

## features
- lookup id by word
- lookup word by id
- lookup id by vec
- count id_to_word keys, word_to_id keys (return same value)

## example
create graph & save

```python
import time
import tensorflow as tf
from models import Word2VecTfModel

NAME = 'default'
VERSION = int(time.time())
SAVE_DIR = f'./tmp/{VERSION}'
SERVING_HOST = 'localhost'
SERVING_PORT = 8500
W2V = {b'apple': [0, 1], b'orange': [2, 3]}

model = Word2VecTfModel(w2v=W2V)

with tf.Session() as sess:
    model.create_graph()
    model.save(save_dir=SAVE_DIR)
```

start tensorflow model server

```bash
tensorflow_model_server \
--port=8500
--model_name='default'
--model_base_path=`pwd`/tmp
```

after tf-serving load model, send request

```python
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2

channel = implementations.insecure_channel(SERVING_HOST, SERVING_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

req = model.create_word_to_id_req([b'apple'])
result_future = stub.Predict.future(req, 1)
result = result_future.result()
ids = tf.make_ndarray(result.outputs['ids']).tolist()

req = model.create_id_to_vec_req(ids)
result_future = stub.Predict.future(req, 1)
result = result_future.result()
print(tf.make_ndarray(result.outputs['vecs']))
```
