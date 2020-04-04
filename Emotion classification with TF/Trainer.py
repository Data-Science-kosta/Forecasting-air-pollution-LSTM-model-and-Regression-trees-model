from sklearn.metrics import classification_report, confusion_matrix
import datetime
import tensorflow as tf
import numpy as np
import os

class Trainer(object):
    def __init__(self, train_images, train_labels, valid_images, valid_labels, model, epochs, batch_size):
        self.model = model
        with self.model.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        self.train_images = train_images #images for training
        self.train_labels = train_labels #labels for training
        self.valid_images = valid_images #images for validation
        self.valid_labels = valid_labels #labels for validation
        self.val_accuracy = 0
        self.train_accuracy = 0
        self.train_loss = 0
        self.val_loss = 0
        self._epochs_training = 0
        self.epochs = epochs
        self.batch_size = batch_size


    def train(self):
        """
        Train the model for self.epochs number of epochs, calling _train_epoch()
        and validate() functions
        """
        # Create new TensorBoard log for each invocation of this function.
        datetime_str = datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')
        # crates tensorboard log
        self.writer_train = tf.summary.FileWriter(logdir=os.path.join(".", "mainLogs", "trainer", datetime_str, "train"), graph=self.model.graph)
        self.writer_val = tf.summary.FileWriter(logdir=os.path.join(".", "mainLogs", "trainer", datetime_str, "val"), graph=self.model.graph)

        print("Training starts")
        while True:
            
            # training part
            loss_train, accuracy_train = self._train_epoch()
            
            print("-------------------\nValidation\n---------------")
            
            #validation part
            loss_val, accuracy_val, _ = self.validate(self.valid_images, self.valid_labels, self.batch_size)
            
            # Compute summaries, and write them to TensorBoard log.
            # Instead of using get_summary, you can return self.model.summary and directly input that in to the add_summary
            summary_train = self.get_summary(loss_train, accuracy_train)
            summary_val = self.get_summary(loss_val, accuracy_val)
            self.writer_train.add_summary(summary_train, self._epochs_training)
            self.writer_val.add_summary(summary_val, self._epochs_training)
            #summary part ends here

            if self._epochs_training == self.epochs:
                print("\n\nTreniranje je zavrseno\n\n")
                break

    def validate(self, valid_images, valid_labels, batch_size):
        """
        Validates the model (ALSO USED FOR TESTING!)
        
        Parameters
        ----------
        valid_images: images
        
        valid_labels: corresponding labels
        
        batch_size: size of one batch
        
        Returns
        -------
        Loss and accuracy computed on the (valid_images,valid_labels)
        """
        print(batch_size)
        with self.model.graph.as_default():
            self.val_accuracy=0
            self.val_loss=0
            # calculate number of batches
            batch_count = int(len(valid_labels) / batch_size)
            # these lists are used for calculating confusion matrix
            pred_list=list()
            actual_list=list()
            for batch_id in range(batch_count):
                batch_start = batch_id * batch_size
                batch_end = min(batch_start + batch_size,len(valid_labels))
                # on the last batch, batch size may not be batch_size
                actual_batch_size = batch_end-batch_start 

                images = valid_images[batch_start:batch_end]
                labels = valid_labels[batch_start:batch_end]

                accuracy, loss, summary, predictions = self.session.run( 
                    # WE HAVENT PROVIDED self.model.optimizer TO THE FETCHES,
                    # SO WE DO NOT DO BACKPROP
                    fetches=(self.model.accuracy, self.model.loss, self.model.
                             summary, self.model.prob
                             ),
                    feed_dict={self.model.images: np.expand_dims(images, 3),
                               self.model.labels: labels
                               }
                    )
                max_pred = np.argmax(predictions, axis=1)
                pred_list.extend(max_pred)
                actual_list.extend(labels)
                self.val_accuracy += accuracy*actual_batch_size
                self.val_loss += loss*actual_batch_size
            self.val_accuracy = self.val_accuracy / len(valid_labels)
            self.val_loss = self.val_loss / len(valid_labels)
    
        print('accuracy in validation: {}'.format(self.val_accuracy))
        actual_list = [ int(x) for x in actual_list ]

        return self.val_loss, self.val_accuracy, confusion_matrix(actual_list, pred_list)
    
    def _train_epoch(self):
        """
        Trains the model for one epoch

        Returns
        -------
        Loss and accuracy on the training set for one epoch
        """
        with self.model.graph.as_default():
            #self.session.run(self.model.reset_accuracy)
            indices1 = np.arange(self.train_labels.shape[0])
            np.random.shuffle(indices1)
            self.train_images=self.train_images[indices1]
            self.train_labels=self.train_labels[indices1]
            self.train_accuracy=0
            self.train_loss=0
            batch_count = int(len(self.train_labels) / self.batch_size) 
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = min(batch_start + self.batch_size,len(self.train_labels))
                # on the last batch, batch size may not be batch_size
                actual_batch_size = batch_end-batch_start
                images = self.train_images[batch_start:batch_end]
                labels = self.train_labels[batch_start:batch_end]
                accuracy, loss, _, summary, prob = self.session.run( # PROVIDING self.model.optimizer MEANS THAT WE DO BACKPROP
                    fetches=(self.model.accuracy, self.model.loss,
                             self.model.optimizer, self.model.summary,
                             self.model.prob
                    ),
                    feed_dict={self.model.images: np.expand_dims(images, 3),
                               self.model.labels: labels
                               }
                    )
                self.train_accuracy+=accuracy*actual_batch_size
                self.train_loss+=loss*actual_batch_size
            self._epochs_training += 1
            self.train_accuracy = self.train_accuracy / len(self.train_labels)
            self.train_loss = self.train_loss / len(self.train_labels)
            # print loss and accuracy for one epoch
            print('\nLOSS:{}'.format(self.train_loss))
            print('\nepoch: {}'.format(self._epochs_training))    
            print('accuracy for epoch:{}: '.format(self.train_accuracy)) 

            return loss, self.train_accuracy  


    def get_summary(self, loss, accuracy):
        """
        Computes summary for given loss and accuracy values.
        
        Parameters
        -----
        loss: The loss value
        
        accuract: The accuracy value
    
        Returns
        -----
        A summary containing given loss and accuracy values as two scalars.
        """
        return tf.Summary(value=[
            tf.Summary.Value(tag="loss", simple_value=loss),
            tf.Summary.Value(tag="accuracy", simple_value=accuracy),
            ])

    def save(self, file_path):
        """
        Saves model parameters to checkpoint file on disk.
        
        Parameters
        -----
        file_path : Path to checkpoint file to be created
    
        Returns
        -----
        None
        """
        with self.model.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.save(self.session, file_path)