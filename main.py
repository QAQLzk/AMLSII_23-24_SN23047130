from functions import *


WORK_DIR = 'datasets/cassava-leaf-disease-classification/'
# ======================================================================================================================
# Data preprocessing
#data split + augmentation(only in train dataset)
data_train_300, data_val_300, data_test_300 = data_preprocessing_and_split(work_dir=WORK_DIR, input_shape=300)
data_train_512, data_val_512, data_test_512 = data_preprocessing_and_split(work_dir=WORK_DIR, input_shape=512)


# Main parameters
BATCH_SIZE = 8
STEPS_PER_EPOCH = len(data_train_300)*0.8 // BATCH_SIZE
VALIDATION_STEPS = len(data_val_300)*0.2 //BATCH_SIZE
# ======================================================================================================================
# Build model for 300 and 512 input shape 
model_300 = build_model(input_shape=300)               # Build model
model_512 = build_model(input_shape=512)               # Build model

# Training of Input shape 300
trained_model_300, best_train_acc_300, best_val_acc_300 =  model_training_plot(
    model= model_300, train_generator= data_train_300, validation_generator= data_val_300,
    EPOCHS=10, BATCH_SIZE=BATCH_SIZE, STEPS_PER_EPOCH=STEPS_PER_EPOCH,VALIDATION_STEPS=VALIDATION_STEPS, input_shape=300
)

# Training of Input shape 512

trained_model_512, best_train_acc_512, best_val_acc_300 =  model_training_plot(
    model= model_512, train_generator= data_train_512, validation_generator= data_val_512,
    EPOCHS=8, BATCH_SIZE=BATCH_SIZE, STEPS_PER_EPOCH=STEPS_PER_EPOCH, VALIDATION_STEPS=VALIDATION_STEPS, input_shape=512
)

# test and evaluate the 512 model
test_acc_512 = test_evaluation(trained_model_512, data_test_512)



# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'