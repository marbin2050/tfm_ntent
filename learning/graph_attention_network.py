from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from spektral.layers import GraphAttention
from preprocessing import create_graph
import numpy as np


class GraphAttentionNetwork:

    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y
        self.pred = None

    def execute(self):

        # create graph
        graph, index_list = create_graph(self.data, only_main_pages=True)

        # create adjacency matrix
        A = graph.get_adjacency_sparse()

        # create masks to select training, validation and test samples
        random_mask = np.random.random(self.x.shape[0])
        train_mask = random_mask < 0.8
        val_mask = (random_mask >= 0.8) * (random_mask < 0.9)
        test_mask = random_mask >= 0.9

        # encode y
        encoded_y = []
        for code in self.y:
            if code == 1:
                encoded_y.append([1, 0, 0])
            elif code == 0:
                encoded_y.append([0, 1, 0])

        y_uncoded = self.y
        self.y = np.array(encoded_y)

        # Parameters
        # channels = 4            # Number of channel in each head of the first GAT layer
        # n_attn_heads = 4        # Number of attention heads in first GAT layer
        # N = X.shape[0]          # Number of nodes in the graph
        N = self.x.shape[0]
        # F = X.shape[1]          # Original size of node features
        F = self.x.shape[1]
        # n_classes = y.shape[1]  # Number of classes
        n_classes = 3
        dropout = 0.6           # Dropout rate for the features and adjacency matrix
        l2_reg = 5e-6           # L2 regularization rate
        learning_rate = 5e-3    # Learning rate
        epochs = 200          # Number of training epochs
        es_patience = 100       # Patience for early stopping

        # Preprocessing operations
        A = A.astype('f4')
        X = self.x.toarray()

        # Model definition
        X_in = Input(shape=(F, ))
        A_in = Input(shape=(N, ), sparse=True)

        graph_attention_2 = GraphAttention(n_classes,
                                           attn_heads=3,
                                           concat_heads=False,
                                           dropout_rate=dropout,
                                           activation='softmax',
                                           kernel_regularizer=l2(l2_reg),
                                           attn_kernel_regularizer=l2(l2_reg)
                                           )([X_in, A_in])

        # Build model
        model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)
        optimizer = Adam(lr=learning_rate)

        metrics = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      weighted_metrics=metrics)

        # model.summary()

        # Train model
        validation_data = ([X, A], self.y, val_mask)

        model.fit([X, A],
                  # y,
                  y=self.y,
                  sample_weight=train_mask,
                  epochs=epochs,
                  batch_size=N,
                  verbose=0,
                  validation_data=validation_data,
                  shuffle=False,  # Shuffling data means shuffling the whole graph
                  callbacks=[
                      EarlyStopping(patience=es_patience, restore_best_weights=True)
                  ])

        # Evaluate model
        # print('Evaluating model.')
        # eval_results = model.evaluate([X, A],
        #                               y=self.y,
        #                               sample_weight=test_mask,
        #                               batch_size=N)

        test_data = ([X, A])

        self.pred = model.predict(test_data, batch_size=N)

        # print('Done.\n'
        #       'Test loss: {}\n'
        #       'Test accuracy: {}'.format(*eval_results))

        # coding predictions
        predictions = []
        for row in self.pred[test_mask]:
            if row[0] >= 0.50:
                predictions.append(1)
            else:
                predictions.append(0)

        predictions = np.array(predictions)

        return predictions, y_uncoded[test_mask]
