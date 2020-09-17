train_data_dir = '/content/chest_xray/chest_xray/train'
val_data_dir = '/content/chest_xray/chest_xray/test'


#-----------------------------#  CNN  #------------------------------# 
model_name = 'model_sept_7'
seed = 42

nb_classes = 3
classes = [ 'COVID-19', #420
            'Normal',   #1311
            'Pneumonia' #4061
            ]

class_dict = {0 : classes[0],
              1 : classes[1],
              2 : classes[2]
              }

class_weights = {0 : 9.6690476,
                 1 : 3.09763539,
                 2 : 1.00
                 }

img_rows, img_cols = 300, 300
batch_size = 32
nb_epoch = 10    


#---------------------------#  CONV2D0 #-----------------------------#                       
nb_filters_0 = 32
kernel_size_0 = (3, 3)
stride_size_0 = (1, 1)
pool_size_0 = (2, 2)
activation_0 = 'swish'
dropout_0 = 0.10
l2_0 = 0.001

#---------------------------#  CONV2D1 #-----------------------------#                       
nb_filters_1 = 64
kernel_size_1 = (3, 3)
stride_size_1 = (1, 1)
pool_size_1 = (2, 2)
activation_1 = 'swish'
dropout_1 = 0.10
l2_1 = 0.001

#---------------------------#  CONV2D2 #-----------------------------#
nb_filters_2 = 64
kernel_size_2 = (3, 3)
stride_size_2 = (1, 1)
pool_size_2 = (2, 2)
activation_2 = 'swish'
dropout_2 = 0.10
l2_2 = 0.001

#---------------------------#  CONV2D3 #-----------------------------#
nb_filters_3 =  128
kernel_size_3 = (3, 3)
stride_size_3 = (2, 2)
pool_size_3 = (2, 2)
activation_3 = 'swish'
dropout_3 = 0.10
l2_3 = 0.001

#---------------------------#  CONV2D4 #-----------------------------#
nb_filters_4 = 128
kernel_size_4 = (3, 3)
stride_size_4 = (2, 2)
pool_size_4 = (2, 2)
activation_4 = 'swish'
dropout_4 = 0.15
l2_4 = 0.001

#---------------------------#  CONV2D5 #-----------------------------#                         
nb_filters_5 = 32
kernel_size_5 = (3, 3)
stride_size_5 = (2, 2)
pool_size_5 = (2, 2)
activation_5 = 'swish'
dropout_5 = 0.20
l2_5 = 0.001

#---------------------------#  DENSE6  #-----------------------------#               
units_6 = 256
activation_6 = 'swish'
dropout_6 = 0.20
l2_6 = 0.001

#---------------------------#  DENSE7  #-----------------------------#            
units_7 = 256
activation_7 = 'swish'
dropout_7 = 0.5
l2_7 = 0.001



#---------------------------#  COMPILE  #----------------------------#                  
compile_optimizer = Adam(lr=0.0001)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_cols, img_rows)
    chanDim = 1
else:
    input_shape = (img_cols, img_rows, 1)
    chanDim = -1



#----------------------------#  MODEL  #-----------------------------# 
model = Sequential()

# #---------------------------#  CONV2D0  #----------------------------# 
# model.add(Conv2D(nb_filters_0,
#                     (kernel_size_0[0], kernel_size_0[1]),
#                     strides=(stride_size_0[0], stride_size_0[1]),
#                     padding='same', activation=activation_0,
#                     input_shape=input_shape))
# model.add(Dropout(dropout_0))

#---------------------------#  CONV2D1  #----------------------------# 
model.add(Conv2D(nb_filters_1,
                    (kernel_size_1[0], kernel_size_1[1]),
                    strides=(stride_size_1[0], stride_size_1[1]),
                    padding='same', #activation=activation_1,
                    kernel_regularizer=regularizers.L2(l2_1),
                    input_shape=input_shape))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_1))
model.add(MaxPooling2D(pool_size=pool_size_1))
model.add(Dropout(dropout_1))

#---------------------------#  CONV2D2  #----------------------------# 
model.add(Conv2D(nb_filters_2,
                    (kernel_size_2[0], kernel_size_2[1]),
                    strides=(stride_size_2[0], stride_size_2[1]),
                    padding='same', #activation=activation_2,
                    kernel_regularizer=regularizers.L2(l2_2)))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_2))
model.add(MaxPooling2D(pool_size=pool_size_2))
model.add(Dropout(dropout_2))

#---------------------------#  CONV2D3  #----------------------------# 
model.add(Conv2D(nb_filters_3,
                    (kernel_size_3[0], kernel_size_3[1]),
                    strides=(stride_size_3[0], stride_size_3[1]),
                    padding='same', #activation=activation_3,
                    kernel_regularizer=regularizers.L2(l2_3)))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_3))
model.add(MaxPooling2D(pool_size=pool_size_3))
model.add(Dropout(dropout_3))

#---------------------------#  CONV2D4  #----------------------------# 
model.add(Conv2D(nb_filters_4,
                    (kernel_size_4[0], kernel_size_4[1]),
                    strides=(stride_size_4[0], stride_size_4[1]),
                    padding='same', #activation=activation_4,
                    kernel_regularizer=regularizers.L2(l2_4)))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_4))
model.add(MaxPooling2D(pool_size=pool_size_4))
model.add(Dropout(dropout_4))

#---------------------------#  CONV2D5  #----------------------------# 
model.add(Conv2D(nb_filters_5,
                    (kernel_size_5[0], kernel_size_5[1]),
                    strides=(stride_size_5[0], stride_size_5[1]),
                    padding='same',# activation=activation_5,
                    kernel_regularizer=regularizers.L2(l2_5)))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_5))
model.add(MaxPooling2D(pool_size=pool_size_5))
model.add(Dropout(dropout_5))


#---------------------------#  FLATTEN  #----------------------------# 
model.add(Flatten())

#---------------------------#  DENSE6  #-----------------------------# 
model.add(Dense(units_6, use_bias=False, 
                kernel_regularizer=regularizers.L2(l2_6)))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_6))
model.add(Dropout(dropout_6))

#---------------------------#  DENSE7  #-----------------------------# 
model.add(Dense(units_7, use_bias=False,
                kernel_regularizer=regularizers.L2(l2_7)))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation(activation_7))
model.add(Dropout(dropout_7))

#---------------------------#  DENSE8  #-----------------------------# 
model.add(Dense(nb_classes))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation('softmax'))

#--------------------------#  METRICS  #-----------------------------# 
METRICS = [ metrics.CategoricalAccuracy(name='ACCURACY'),
            metrics.AUC(name='AUC',curve='pr', multi_label=True),
           metrics.SensitivityAtSpecificity(0.5, name='Sens@Spec'),
           metrics.SpecificityAtSensitivity(0.5, name='Spec@Sens')]

#--------------------------#  COMPILE  #-----------------------------# 
model.compile(loss='categorical_crossentropy',
                optimizer=compile_optimizer,
                metrics=METRICS)


#----------------------#  DATAGENERATORS  #--------------------------# 
def AHE(img):
    ahe = exposure.equalize_adapthist(img, clip_limit=0.03)
    return ahe

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    preprocessing_function=AHE
    )

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.01,
    zoom_range=0.01,
    preprocessing_function=AHE
    )


#-------------------------#  GENERATORS  #---------------------------# 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_cols, img_rows),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)


#---------------------------#  SUMMARY  #-----------------------------# 
model.summary()


es = EarlyStopping(monitor='val_loss', mode='min', 
                   verbose=1, patience=15,
                   restore_best_weights=True)

name = model_name + '.h5'
mc = ModelCheckpoint(name, monitor='val_loss', 
                     mode='min', verbose=1, save_best_only=True)



#-----------------------------#  FIT  #-------------------------------# 
history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // batch_size,
    epochs=nb_epoch,
    validation_data= validation_generator,
    validation_steps= validation_generator.samples // batch_size,
    class_weight=class_weights,
    callbacks=[es]
    )

label_map = (train_generator.class_indices)
print(label_map)


#ACCURACY---------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12,6))
ax[0][0].plot(history.history['ACCURACY'])
ax[0][0].plot(history.history['val_ACCURACY'])
ax[0][0].set_ylim(0, 1.0)
ax[0][0].set_title('Accuracy')
ax[0][0].set_ylabel('Accuracy')
ax[0][0].set_xlabel('Epoch')
ax[0][0].legend(['train', 'val'], loc='best')
#LOSS-------------------------------------------
ax[0][1].plot(history.history['loss'])
ax[0][1].plot(history.history['val_loss'])
ax[0][1].set_ylim(0)
ax[0][1].set_title('Loss')
ax[0][1].set_ylabel('Loss')
ax[0][1].set_xlabel('Epoch')
ax[0][1].legend(['train', 'val'], loc='best')
plt.tight_layout()
#Sens@Spec---------------------------------------
ax[1][0].plot(history.history['Sens@Spec'])
ax[1][0].plot(history.history['val_Sens@Spec'])
ax[1][0].set_ylim(0, 1.0)
ax[1][0].set_title('Sens@Spec')
ax[1][0].set_ylabel('Sens@Spec')
ax[1][0].set_xlabel('Epoch')
ax[1][0].legend(['train', 'val'], loc='best')
#LOSS-------------------------------------------
ax[1][1].plot(history.history['Spec@Sens'])
ax[1][1].plot(history.history['val_Spec@Sens'])
ax[1][1].set_ylim(0)
ax[1][1].set_title('Spec@Sens')
ax[1][1].set_ylabel('Spec@Sens')
ax[1][1].set_xlabel('Epoch')
ax[1][1].legend(['train', 'val'], loc='best')
plt.tight_layout()

# plot_model(model, to_file='image2.png', show_shapes=True, show_layer_names=True)
;


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
  
    plt.figure(figsize = (10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.suptitle('Confusion matrix', fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=40)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(validation_generator.classes, y_pred)
plot_confusion_matrix(cm, classes, normalize=False)


def find_true_class(file_path):
    true_class = None
    if 'CORONA' in file_path:
        true_class = 'COVID-19'
    elif 'PNEUMONIA' in file_path:
        true_class = 'Pneumonia'
    elif 'NORMAL' in file_path:
        true_class = 'Normal'
    return true_class

def prediction(file_path, ax, model):
    test_image = cv2.imread(file_path)
    test_image = cv2.resize(test_image, (img_rows,img_cols), interpolation=cv2.INTER_NEAREST)
    ax.imshow(test_image)
    
    ax.plot(1,1, 'red')
    ax.set_xticks([],[])
    ax.set_yticks([],[])

    image = load_img(file_path,color_mode='grayscale', target_size=(img_rows,img_cols))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr / 255
    probs = model.predict(input_arr)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]

    true_class = find_true_class(file_path)

    if true_class is not None:
        ax.set_title(f"Pred: {pred_class} \nActual: {true_class}", fontsize=13)



image1 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg'
image2 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg'
image3 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg'
image4 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1436-0001.jpeg'
image5 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1437-0001.jpeg'
image6 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1438-0001.jpeg'
image7 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1440-0001.jpeg'
image8 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg'
image9 = '/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg'

fig,ax = plt.subplots(3,3,figsize=(12,12))
fig.suptitle('Normal', fontsize=18)
prediction(image1, ax[0][0], model)
prediction(image2, ax[0][1], model)
prediction(image3, ax[0][2], model)
prediction(image4, ax[1][0], model)
prediction(image5, ax[1][1], model)
prediction(image6, ax[1][2], model)
prediction(image7, ax[2][0], model)
prediction(image8, ax[2][1], model)
prediction(image9, ax[2][2], model)


