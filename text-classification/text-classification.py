import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

# Initial data and methods

debug = False


def plotGraphs(history, metric):
    """
    Display graph of history and metric
    """
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val' + metric])


def waitUser():
    """
    Use the input keyboard method to wait user to continue
    """
    if debug:
        input('Presiona enter para continuar')


dataset, info = tfds.load(
    'imdb_reviews', with_info=True, as_supervised=True)
trainDataset, testDataset = dataset['train'], dataset['test']
print(trainDataset.element_spec)
waitUser()

for example, label in trainDataset.take(1):
    print('text: ', example.numpy())
    print('label: ', label.numpy())
waitUser()

# Shuffle data
BUFFER_SIZE = 10000
BATCH_SIZE = 64

trainDataset = trainDataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
testDataset = testDataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for example, label in trainDataset.take(1):
    print('texts: ', example.numpy()[:3])
    print()
    print('labels: ', label.numpy()[:3])
waitUser()

# Create the text encoder
VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(trainDataset.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())
print(vocab[:20])
waitUser()

for example, label in trainDataset.take(1):
    encodedExample = encoder(example)[:3].numpy()
    print(encodedExample)
waitUser()

for example, label in trainDataset.take(1):
    encodedExample = encoder(example)[:3].numpy()
    for n in range(3):
        print('Original: ', example[n].numpy())
        print('Round-trip: ', ' '.join(vocab[encodedExample[n]]))
        print()
waitUser()

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1),
])

print([layer.supports_masking for layer in model.layers])

# predict on a sample text without padding.
sampleText = ('The movie was cool. The animation and the graphics '
              'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sampleText]))
print(predictions[0])

# predict on a sample text with padding
padding = "the " * 2000
predictions = model.predict(np.array([sampleText, padding]))
print(predictions[0])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']
              )
model.summary()
waitUser()

# Train model
history = model.fit(trainDataset, epochs=10,
                    validation_data=testDataset, validation_steps=30)

testLoss, testAcc = model.evaluate(testDataset)
print('Test Loss: ', testLoss)
print('Test Accuracy: ', testAcc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plotGraphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plotGraphs(history, 'loss')
plt.ylim(0, None)

sampleText = ('The movie was cool. The animation and the graphics '
              'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sampleText]))

model.save('classification-model', save_format='tf')
