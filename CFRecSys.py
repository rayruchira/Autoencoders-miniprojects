#import required libraries
import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

#getting required files
r_df = pd.read_csv('/home/ray/Projects/datasets/train_100k.csv', delimiter = ',')
r=np.array(r_df, dtype=int)[:,1:]

r_t_df = pd.read_csv('/home/ray/Projects/datasets/test.csv', delimiter = ',')
r_t=np.array(r_t_df, dtype=int)[:,1:]

#checking if it can acces files
print(r_t[0])
#getting max no of movies and users
nb_users = int(max(max(r[:, 0]), max(r_t[:, 0])))
nb_movies = int(max(max(r[:, 1]), max(r_t[:, 1])))

# printing the number of users and movies
print(f'The number of users {nb_users}, The number of movies {nb_movies}')

#Stacked Autoencoder

class StackedAutoEncoder:
    def __init__(self,x ):
        super(StackedAutoEncoder, self).__init__() # getting all the functionality from the parent class
        self.x=x

    #DATA PREPROCESSING
    def convert_fn(self, ):
        converted_data = []
        data=self.x
        for user_id in range(1, nb_users + 1):
            #getting all the movies ids that rated by every user
            movies_for_user_id = data[:, 1][data[:, 0] == user_id] # getting the movies ids that taken be the current user in the for loop
            rating_for_user_id = data[:, 2][data[:, 0] == user_id]
            
            ratings = np.zeros(nb_movies) # initialze all the ratings with zeros then include the rated movies
            ratings[movies_for_user_id - 1] = rating_for_user_id
            converted_data.append(list(ratings))
            
        self.x=np.array(converted_data)
    
    #encoder layers
    def _encoder(self):
        inputs = tf.keras.layers.Input(shape=(nb_movies,))
        enc1 = tf.keras.layers.Dense(30, activation='relu')(inputs)
        enc2= tf.keras.layers.Dense(15,activation='relu')(enc1)
        enc3= tf.keras.layers.Dense(10,activation='relu')(enc2)
        model = tf.keras.Model(inputs, enc3)
        self.encoder = model
        return model
    
    #decoder layers
    def _decoder(self):
        inputs = tf.keras.layers.Input(shape=(10,))
        dec1= tf.keras.layers.Dense(15)(inputs)
        dec2=tf.keras.layers.Dense(30)(dec1)
        dec3= tf.keras.layers.Dense(nb_movies)(dec2)
        model = tf.keras.Model(inputs, dec3)
        self.decoder = model
        return model

    #autoencoder model
    def encoder_decoder(self):
        
        ec = self._encoder()
        dc = self._decoder()
        
        inputs = tf.keras.layers.Input(shape=(nb_movies,))
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = tf.keras.Model(inputs, dc_out)
        
        self.model = model
        return model

    #training 
    def fit(self, batch_size=32, epochs=300):
        #optimizer define
        optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=0.01, decay=0.5)
        
        #model compilation
        self.model.compile(optimizer=optimizer, loss='mse',metrics=['accuracy'])
        log_dir = './log/'
        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[tbCallBack])
    #saving weights
    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weightsAEM.h5')
            self.decoder.save(r'./weights/decoder_weightsAEM.h5')
            self.model.save(r'./weights/ae_weightsAEM.h5')

if __name__ == '__main__':
   
    ae = StackedAutoEncoder(x=r)
    ae.convert_fn()
    print(f'\nThe dimensions of the data is {len(ae.x)} X {len(ae.x[0])}')

    encoder=ae.encoder_decoder()
    ae.fit(batch_size=32, epochs=50)
    ae.save()

    #testing
    ae = StackedAutoEncoder(x=r_t)
    ae.convert_fn()
    inputs=ae.x

    #loading weights
    model=tf.keras.models.load_model(r'./weights/ae_weightsAEM.h5')

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(inputs, inputs, batch_size=64)
    print('test loss, test acc:', results)
    
