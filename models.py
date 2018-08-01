import tensorflow as tf
from tensorflow_serving.apis import predict_pb2


def create_req_base(name, version):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    if version is not None:
        version = wrappers_pb2.Int64Value()
        version.value = VERSION
        request.model_spec.version.CopyFrom(version)
    return request


def create_consts(w2v):
    _words = []
    _ids = []
    _vecs = []
    for id, (word, vec) in enumerate(w2v.items()):
        _words.append(word)
        _ids.append(id)
        _vecs.append(vec)

    words = tf.constant(_words, dtype=tf.string)
    ids = tf.constant(_ids, dtype=tf.int64)
    vecs = tf.constant(_vecs, dtype=tf.float32)
    return words, ids, vecs


class SimpleWord2VecTfModel:
    def __init__(self, w2v):
        self.w2v = w2v

    def create_word_to_vec_req(self, words, name='default', version=None):
        request = create_req_base(name=name, version=version)
        request.model_spec.signature_name = 'word_to_vec'
        request.inputs['words'].CopyFrom(tf.contrib.util.make_tensor_proto(words, shape=[len(words),], dtype=tf.string))
        return request

    def create_graph(self):
        self.words_placeholder = tf.placeholder(tf.string, shape=[None, ])
        words, ids, vecs = create_consts(self.w2v)

        word_to_id_table = tf.contrib.lookup.HashTable(
            initializer=tf.contrib.lookup.KeyValueTensorInitializer(
                keys=words,
                values=ids,
                key_dtype=tf.string,
                value_dtype=tf.int64,
                name='word_to_id_key_value_initializer',
            ),
            default_value=self.id_default_value,
            name='word_to_id_table',
        )

        self.word_to_id = word_to_id_table.lookup(
            keys=self.words_placeholder,
            name='word_to_id_lookup',
        )
        self.word_to_vec = tf.nn.embedding_lookup(
            params=vecs,
            ids=ids,
            name='word_to_vec_lookup',
        )

    def save(self, save_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
        with tf.Session() as sess:
            self.create_graph()

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'word_to_vec': tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            'words': tf.saved_model.utils.build_tensor_info(self.words_placeholder),
                        },
                        outputs={
                            'vecs': tf.saved_model.utils.build_tensor_info(self.word_to_vec),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    ),
                },
                legacy_init_op=tf.saved_model.main_op.main_op(),
            )
        builder.save()


class Word2VecTfModel:
    id_default_value = -1
    word_default_value = b'UNK'

    def __init__(self, w2v):
        self.w2v = w2v

    def create_req_base(self, name, version):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = name
        if version is not None:
            version = wrappers_pb2.Int64Value()
            version.value = VERSION
            request.model_spec.version.CopyFrom(version)

    def create_word_to_id_req(self, words, name='default', version=None):
        request = create_req_base(name=name, version=version)
        request.model_spec.signature_name = 'word_to_id'
        request.inputs['words'].CopyFrom(tf.contrib.util.make_tensor_proto(words, shape=[len(words),], dtype=tf.string))
        return request

    def create_id_to_word_req(self, ids, name='default', version=None):
        request = create_req_base(name=name, version=version)
        request.model_spec.signature_name = 'id_to_word'
        request.inputs['ids'].CopyFrom(tf.contrib.util.make_tensor_proto(ids, shape=[len(ids),], dtype=tf.int64))
        return request

    def create_id_to_word_size_req(self, name='default', version=None):
        request = create_req_base(name=name, version=version)
        request.model_spec.signature_name = 'id_to_word_size'
        return request

    def create_word_to_id_size_req(self, name='default', version=None):
        request = create_req_base(name=name, version=version)
        request.model_spec.signature_name = 'word_to_id_size'
        return request

    def create_id_to_vec_req(self, ids, name='default', version=None):
        request = create_req_base(name=name, version=version)
        request.model_spec.signature_name = 'id_to_vec'
        request.inputs['ids'].CopyFrom(tf.contrib.util.make_tensor_proto(ids, shape=[len(ids),], dtype=tf.int64))
        return request

    def create_graph(self):
        self.words_placeholder = tf.placeholder(tf.string, shape=[None, ])
        self.ids_placeholder = tf.placeholder(tf.int64, shape=[None, ])
        words, ids, vecs = create_consts(self.w2v)

        word_to_id_table = tf.contrib.lookup.HashTable(
            initializer=tf.contrib.lookup.KeyValueTensorInitializer(
                keys=words,
                values=ids,
                key_dtype=tf.string,
                value_dtype=tf.int64,
                name='word_to_id_key_value_initializer',
            ),
            default_value=self.id_default_value,
            name='word_to_id_table',
        )
        id_to_word_table = tf.contrib.lookup.HashTable(
            initializer=tf.contrib.lookup.KeyValueTensorInitializer(
                keys=ids,
                values=words,
                key_dtype=tf.int64,
                value_dtype=tf.string,
                name='id_to_word_key_value_initializer',
            ),
            default_value=self.word_default_value,
            name='id_to_word_table',
        )

        self.word_to_id = word_to_id_table.lookup(
            keys=self.words_placeholder,
            name='word_to_id_lookup',
        )
        self.word_to_id_size = word_to_id_table.size(
            name='word_to_id_size')

        self.id_to_word = id_to_word_table.lookup(
            keys=self.ids_placeholder,
            name='id_to_word_lookup',
        )
        self.id_to_word_size = id_to_word_table.size(
            name='id_to_word_size')

        self.id_to_vec = tf.nn.embedding_lookup(
            params=vecs,
            ids=self.ids_placeholder,
            name='id_to_vec_lookup',
        )

    def save(self, save_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
        with tf.Session() as sess:
            self.create_graph()

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'word_to_id': tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            'words': tf.saved_model.utils.build_tensor_info(self.words_placeholder),
                        },
                        outputs={
                            'ids': tf.saved_model.utils.build_tensor_info(self.word_to_id),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    ),
                    'word_to_id_size': tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={},
                        outputs={
                            'size': tf.saved_model.utils.build_tensor_info(self.word_to_id_size),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    ),
                    'id_to_word': tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            'ids': tf.saved_model.utils.build_tensor_info(self.ids_placeholder),
                        },
                        outputs={
                            'words': tf.saved_model.utils.build_tensor_info(self.id_to_word),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    ),
                    'id_to_word_size': tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={},
                        outputs={
                            'size': tf.saved_model.utils.build_tensor_info(self.id_to_word_size),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    ),
                    'id_to_vec': tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            'ids': tf.saved_model.utils.build_tensor_info(self.ids_placeholder),
                        },
                        outputs={
                            'vecs': tf.saved_model.utils.build_tensor_info(self.id_to_vec),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    ),
                },
                legacy_init_op=tf.saved_model.main_op.main_op(),
            )
        builder.save()


if __name__ == '__main__':
    from functools import reduce
    import numpy as np

    w2v = {b'apple': [0, 1], b'orange': [2, 3]}
    model = Word2VecTfModel(w2v)

    with tf.Session() as sess:
        model.create_graph()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        words = list(w2v.keys())
        _ids = sess.run(model.word_to_id, feed_dict={model.words_placeholder: words})
        _words = sess.run(model.id_to_word, feed_dict={model.ids_placeholder: _ids})

        def cmp(prev, cur):
            first = cur[0]
            second = cur[1]
            return False if first != second else prev

        assert(reduce(cmp, zip(words, _words), True))
        assert(sess.run(model.word_to_id_size) == len(w2v))
        assert(sess.run(model.id_to_word_size) == len(w2v))

        assert(sess.run(model.word_to_id, feed_dict={model.words_placeholder: [b'pinapple']})[0] == model.id_default_value)
        assert(sess.run(model.id_to_word, feed_dict={model.ids_placeholder: [len(w2v) + 1]})[0] == model.word_default_value)

        def cmp_vecs(prev, cur):
            vec = cur[0]
            _vec = cur[1]
            return reduce(cmp, zip(vec, _vec), True)

        _vecs = sess.run(model.id_to_vec, feed_dict={model.ids_placeholder: _ids})
        vecs = list(w2v.values())
        assert(reduce(cmp_vecs, zip(vecs, _vecs), True))
