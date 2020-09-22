import logging
logging.basicConfig(level=logging.ERROR)
from transformers import TFBertPreTrainedModel,TFBertMainLayer,BertTokenizer
import tensorflow as tf
from transformers.modeling_tf_utils import (
    TFQuestionAnsweringLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    shape_list,
)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def convert_example_to_feature(review):
  
  # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
    return tokenizer.encode_plus(review, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True
              )
# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    
    for (i, row) in enumerate(ds.values):
#     for index, row in ds.iterrows():
#         review = row["text"]
#         label = row["y"]
        review = row[1]
        label = list(row[2:])
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)



    
class TFBertForMultilabelClassification(TFBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultilabelClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier',
                                                activation='sigmoid')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)



def measure_auc(label,pred):
  auc = [roc_auc_score(label[:,i],pred[:,i]) for i in list(range(6))]
  return pd.DataFrame({"label_name":["toxic","severe_toxic","obscene","threat","insult","identity_hate"],"auc":auc})


if __name__ == '__main__': 

    # parameters
    train_path = "train.csv" # 数据路径
    test_path = "test.csv" # 数据路径
    model_path = "bert-base-uncased" #模型路径，建议预先下载(https://huggingface.co/bert-base-uncased#)

    # parameters
    max_length = 128
    batch_size = 32
    learning_rate = 2e-5
    number_of_epochs = 2
    num_classes = 6 # 类别数

    # read data
    train_val_data = pd.read_csv(train_path)
    TRAIN_VAL_RATIO = 0.9
    LEN = train_val_data.shape[0]
    SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)
    # train data
    train_data = train_val_data[:SIZE_TRAIN]
    # val data
    val_data = train_val_data[SIZE_TRAIN:]
    # test data
    test_data = pd.read_csv(test_path)
    
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # train dataset
    ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(batch_size)
    # val dataset
    ds_val_encoded = encode_examples(val_data).batch(batch_size)
    # test dataset
    ds_test_encoded = encode_examples(test_data).batch(batch_size)
    
    # model initialization
    model = TFBertForMultilabelClassification.from_pretrained(model_path, num_labels=num_classes)
    # optimizer Adam recommended
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # fit model
    bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
    # evaluate val_set
    pred=model.predict(ds_val_encoded)[0]
    df_auc = measure_auc(val_data.iloc[:,2:].astype(np.float32).values,pred)
    print("val set mean column auc:",df_auc)
    #predict test_set
