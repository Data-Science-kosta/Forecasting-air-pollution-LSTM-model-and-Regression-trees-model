import tensorflow as tf

class Predictor:
    def __init__(self, model,RESOLUTION):
        self.resolution = RESOLUTION
        self.model = model
        with self.model.graph.as_default():
            self.session = tf.Session()

    def predict(self, image):
        """
        Makes a prediction on image, outputs the probabilities of the labels
        
        Parameters
        -----
        image : image
        
        Returns
        -----
        logits: probabilities, output of the softamx layer
        """
        return self.session.run(
            fetches=self.model.prob,
            feed_dict={
                self.model.images:image.reshape(1,self.resolution,self.resolution,1)
                }
            )
    def restore(self,path):
        """
        Restores pretrained weights from the path
        
        Parameters
        -----
        path : path to the weights
        Returns
        -----
        None
        """
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, path)