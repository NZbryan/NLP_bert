# tensorflow 2.0+ 基于BERT模型的文本分类

### 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC](https://link.zhihu.com/?target=http%3A//thuctc.thunlp.org/)：一个高效的中文文本分类工具包下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类2W条数据。

类别如下：

```python
财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐
```

数据集在 [data.txt](https://github.com/NZbryan/MachineLearning/blob/master/NLP/data.txt)

现将数据集按照层次抽样划分为训练集、验证集、测试集：

| 数据集 | 数据量 |
| ------ | ------ |
| 训练集 | 18万   |
| 验证集 | 1万    |
| 测试集 | 1万    |

```python
from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(df):
    train_set, x = train_test_split(df, 
        stratify=df['label'],
        test_size=0.1, 
        random_state=42)
    val_set, test_set = train_test_split(x, 
        stratify=x['label'],
        test_size=0.5, 
        random_state=43)

    return train_set,val_set, test_set


df_raw = pd.read_csv("data.txt",sep="\t",header=None,names=["text","label"])    
# label
df_label = pd.DataFrame({"label":["财经","房产","股票","教育","科技","社会","时政","体育","游戏","娱乐"],"y":list(range(10))})
df_raw = pd.merge(df_raw,df_label,on="label",how="left")

train_data,val_data, test_data = split_dataset(df_raw)
```







### 使用TensorFlow 2.0+ keras API微调BERT

现在，我们需要在所有样本中应用 BERT  tokenizer 。我们将token映射到词嵌入。这可以通过encode_plus完成。

```python
def convert_example_to_feature(review):
    return tokenizer.encode_plus(review, 
                                 add_special_tokens = True, # add [CLS], [SEP]
                                 max_length = max_length, # max length of the text that can go to BERT
                                 pad_to_max_length = True, # add [PAD] tokens
                                 return_attention_mask = True, # add attention mask to not focus on pad tokens
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
    
    for index, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)
```



我们可以使用以下函数对数据集进行编码：

```python
# train dataset
ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(batch_size)
# val dataset
ds_val_encoded = encode_examples(val_data).batch(batch_size)
# test dataset
ds_test_encoded = encode_examples(test_data).batch(batch_size)
```



创建模型

```python
from transformers import TFBertForSequenceClassification
import tensorflow as tf

model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=10)
```

编译与训练模型

```python
# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
learning_rate = 2e-5
# we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
number_of_epochs = 8

# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=10)

# optimizer Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# fit model
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
# evaluate test set
model.evaluate(ds_test_encoded)
```



以下是8个epochs的训练结果:

```python
Epoch 1/8
1407/1407 [==============================] - 2012s 1s/step - loss: 1.5890 - accuracy: 0.8952 - val_loss: 1.5220 - val_accuracy: 0.9298
Epoch 2/8
1407/1407 [==============================] - 1998s 1s/step - loss: 1.5114 - accuracy: 0.9390 - val_loss: 1.5133 - val_accuracy: 0.9317
Epoch 3/8
1407/1407 [==============================] - 2003s 1s/step - loss: 1.4998 - accuracy: 0.9487 - val_loss: 1.5126 - val_accuracy: 0.9331
Epoch 4/8
1407/1407 [==============================] - 1995s 1s/step - loss: 1.4941 - accuracy: 0.9563 - val_loss: 1.5090 - val_accuracy: 0.9369
Epoch 5/8
1407/1407 [==============================] - 1998s 1s/step - loss: 1.4901 - accuracy: 0.9612 - val_loss: 1.5099 - val_accuracy: 0.9367
Epoch 6/8
1407/1407 [==============================] - 1995s 1s/step - loss: 1.4876 - accuracy: 0.9641 - val_loss: 1.5104 - val_accuracy: 0.9346
Epoch 7/8
1407/1407 [==============================] - 1994s 1s/step - loss: 1.4859 - accuracy: 0.9668 - val_loss: 1.5104 - val_accuracy: 0.9356
Epoch 8/8
1407/1407 [==============================] - 1999s 1s/step - loss: 1.4845 - accuracy: 0.9688 - val_loss: 1.5114 - val_accuracy: 0.9321
                
79/79 [==============================] - 37s 472ms/step - loss: 1.5037 - accuracy: 0.9437
[1.5037099123001099, 0.9437000155448914]
```



可以看到，训练集正确率96.88%，验证集正确率93.21%，测试集上正确率94.37%。





### 运行环境

```shell
linux: CentOS Linux release 7.6.1810

python: Python 3.6.10

packages:

tensorflow==2.3.0
transformers==3.02
pandas==1.1.0
scikit-learn==0.22.2
```



### 使用方式

```shell
git clone https://github.com/NZbryan/NLP_bert.git
cd NLP_bert

python3 tf2.0_bert_emb_ch_MultiClass.py
```


由于数据量较大,训练时间长,建议在GPU下运行,或者到colab去跑。



参考：[Text classification with transformers in Tensorflow 2: BERT](https://medium.com/atheros/text-classification-with-transformers-in-tensorflow-2-bert-2f4f16eff5ad)
