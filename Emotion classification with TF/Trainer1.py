import datetime
import tensorflow as tf
import numpy as np
import os
# CONCATENATES TWO DATASETS
class Trainer1(object):
    def __init__(self, train_images, train_labels, valid_images, valid_labels,train_images1, train_labels1, valid_images1, valid_labels1, model, epochs, batch_size):
        self.model = model
        with self.model.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        # CK dataset
        self.train_images = train_images #images for training
        self.train_labels = train_labels #labels for training
        self.valid_images = valid_images #images for validation
        self.valid_labels = valid_labels #labels for validation
        # FER dataset
        self.train_images1 = train_images1 #images1 for training
        self.train_labels1 = train_labels1 #labels1 for training
        self.valid_images1 = valid_images1 #images1 for validation
        self.valid_labels1 = valid_labels1 #labels1 for validation

        self.val_accuracy = 0
        self.train_accuracy = 0
        self.train_loss = 0
        self.val_loss = 0
        self._epochs_training = 0
        self.epochs = epochs
        self.loss=0
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

        print("\nTraining starts")
        k=0
        while True:   
            # training part
            
            loss_train, accuracy_train, k = self._train_epoch(k)
            print("-------------------\nValidation\n-------------------")
            print ("k={}".format(k))
            #validation part
            loss_val, accuracy_val, k = self.validate(self.valid_images, self.valid_labels, self.batch_size,k)
            
            # Compute summaries, and write them to TensorBoard log.
            summary_train = self.get_summary(loss_train, accuracy_train)
            summary_val = self.get_summary(loss_val, accuracy_val)
            self.writer_train.add_summary(summary_train, self._epochs_training)
            self.writer_val.add_summary(summary_val, self._epochs_training)
            #summary part ends here
            print ("k={}".format(k))
            if self._epochs_training == self.epochs:
                print("\n\nTreniranje je zavrseno\n\n")
                break

    def validate(self, valid_images, valid_labels, batch_size,k=25026):
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
        k: index where starts next set of images in FER dataset
        """
        with self.model.graph.as_default():
            self.val_accuracy = 0
            self.val_loss = 0
            #self.session.run(self.model.reset_accuracy)
            dim=np.shape(valid_images)[0]
            validate_images=np.concatenate((self.valid_images,self.valid_images1[k:k+dim,:,:]),axis=0)
            validate_labels=np.concatenate((self.valid_labels,self.valid_labels1[k:k+dim]),axis=0)
            batch_count = int(len(valid_labels) / batch_size)
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = min(batch_start + self.batch_size,len(valid_labels))
                # on the last batch, batch size may not be batch_size
                actual_batch_size = batch_end-batch_start

                images = valid_images[batch_start:batch_end]
                labels = valid_labels[batch_start:batch_end]

                accuracy, loss, summary, predictions = self.session.run(
                    (self.model.accuracy, self.model.loss, self.model.summary, (self.model.guess_class, self.model.guess_prob)),
                    feed_dict={self.model.images: np.expand_dims(images, 3),
                               self.model.labels: labels})
                self.val_accuracy += accuracy*actual_batch_size
                self.val_loss += loss*actual_batch_size
            self.val_accuracy = self.val_accuracy / len(valid_labels)
            self.val_loss = self.val_loss / len(valid_labels)
    
        print('accuracy in validation: {}'.format(self.val_accuracy))

        return self.val_loss, self.val_accuracy, k+dim
    
    def _train_epoch(self, k):
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
            training_images=np.concatenate((self.train_images, self.train_images1[k:k+len(indices1),:,:]), axis=0)
            training_labels=np.concatenate((self.train_labels, self.train_labels1[k:k+len(indices1)]), axis=0)
            batch_count = int(len(training_labels) / self.batch_size) 
            for batch_id in range(batch_count):
                batch_start = batch_id * self.batch_size
                batch_end = min(batch_start + self.batch_size,len(training_labels))
                actual_batch_size = batch_end-batch_start
                images = training_images[batch_start:batch_end]
                labels = training_labels[batch_start:batch_end]
                accuracy, loss, _, summary, prob = self.session.run(
                    (self.model.accuracy, self.model.loss, self.model.optimizer, self.model.summary, self.model.prob),
                    feed_dict={self.model.images: np.expand_dims(images, 3),self.model.labels: labels})
                self.train_accuracy+=accuracy*actual_batch_size
                self.train_loss+=loss*actual_batch_size
            self._epochs_training += 1
            self.train_accuracy=self.train_accuracy / len(training_labels)
            self.train_loss=self.train_loss / len(training_labels)
            print('\nLOSS:{}'.format(self.train_loss))
            print('\nepoch: {}'.format(self._epochs_training))    
            print('accuracy for epoch:{}: '.format(self.train_accuracy)) 
            return self.train_loss, self.train_accuracy, k+len(indices1)


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