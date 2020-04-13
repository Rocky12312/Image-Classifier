#Basically we need to select good hyperparameters while building a model so as to get the minimum loss and good output predictions
#We can use KerasTuner for that(for optimizing the performance of the model)
#Importing the necessary libraries we need for creating the optimizer
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from keras.engine.hyperparameters import HyperParameters

#Function which will help in finding the best parameters(hyperparameters more than one, so that we can train model on each of them and see the performance and get the best set of hyperparameters)
def build_model(hp):
    #Adding the sequential layer
    model = keras.Sequential()
    #Adding the convolutional layers
    model.add(keras.layers.Conv2D(filters =        hp.Int("conv_2_filter",min_value=32,max_value=64,step=16),
kernel_size=hp.choice("conv_2_kernel",value=[3,5]),
activation = "relu)
    #Adding the layers(Dense layers)
    model.add(layers.Dense(units=hp.Int("units",
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation="relu"))
    #Flattenning the output
    model.add(keras.layers.Flatten())
    #Adding the final output layer having 6 categories(in our case)
    model.add(layers.Dense(6, activation="softmax"))
    #Compiling the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate",
                      values=[1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


tuner_search = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    #No of executions per trial(say 3 epochs per hyperparameter set)
    executions_per_trial=3,
    directory="dir",
    project_name="xyz")
#Running the search on all the set of hyperparameters
tuner_search.search("train_data","train_data_labels",epochs=20,validation_split=0.2)

#Getting the model with best set of parameters and now we using this model and training it on the data
model = tuner_search.get_best_models(num_models = 1)[0]
model.summary()
#Training the model(the model with best hyperparameters out of all we get) 
model.fit("training_data","training_labels",epochs = 20,validation_split=0.1)
#Also if want to save model
model.save("abc.h5")
