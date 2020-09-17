img_rows, img_cols = 300, 300

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    chanDim = 1
else:
    input_shape = (img_rows, img_cols, 1)
    chanDim = -1

epochs = 7000
batch_size = 8

def AHE(img):
  img = img / 255
  img = exposure.equalize_adapthist(img, clip_limit=0.03)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

def images_resize(path, img_rows, img_cols):
lst = [os.path.join(path, i) for i in os.listdir(path)]
return [cv2.resize(cv2.imread(path), (img_rows, img_cols)) for path in lst]


image_dir = "/content/JSRT/JSRT/"
label_dir = "/content/BSE_JSRT/BSE_JSRT/"

x_data = images_resize(image_dir, img_rows, img_cols)
y_data = images_resize(label_dir, img_rows, img_cols)

x_images = np.array([i for i in x_data]).reshape(-1,img_rows,img_cols,1)
y_images = np.array([i for i in y_data]).reshape(-1,img_rows,img_cols,1)


autoencoder = Sequential([
            Conv2D(16, (3, 3), padding = 'same', input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2,2), padding = 'same'),

            Conv2D(32, (3, 3), padding = 'same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2,2), padding = 'same'),

            Conv2D(64, (3, 3), padding = 'same'),
            BatchNormalization(),
            Activation('relu'),
            # MaxPooling2D((2,2), padding = 'same'),

            Conv2D(128, (3, 3), padding = 'same'),
            BatchNormalization(),
            Activation('linear'),
            # MaxPooling2D((2,2), padding = 'same'),
            
            Flatten(),
            Reshape((75, 75, 128)),

            Conv2D(128, (3, 3), padding = 'same'),
            BatchNormalization(),
            Activation('relu'),

            Conv2DTranspose(64, kernel_size=2, strides=2, padding="same"),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(32, (3, 3), padding = 'same'),
            BatchNormalization(),
            Activation('relu'),

            Conv2DTranspose(16, kernel_size=2, strides=2, padding="same"),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(8, (3, 3), padding = 'same'),
            BatchNormalization(),
            Activation('relu'),

            # Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu"),
            # Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu"),
            Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')
            ])

# input_img = Input(shape=input_shape)
# x = Conv2D(16, kernel_size=3, padding="SAME", activation="relu")(input_img)
# x = BatchNormalization()(x)
# x = Conv2D(32, kernel_size=3, padding="SAME", activation="relu")(input_img)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=2)(x)
# x = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = Conv2D(128, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=2)(x)
# x = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# # x = MaxPooling2D(pool_size=2)(x)
# x = Conv2DTranspose(256, kernel_size=3, strides=2, padding="SAME", activation="relu",
#                                  input_shape=[3, 3, 256])(x)
# x = BatchNormalization()(x)
# x = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = Conv2D(32, kernel_size=3, padding="SAME", activation="relu")(x)
# x = BatchNormalization()(x)
# x = Conv2D(1, kernel_size=3, padding="SAME", activation="linear")(x)
# # x = Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid")(x)

# autoencoder = Model(input_img, x)
autoencoder.summary()


METRICS = [metrics.RootMeanSquaredError(name='RMSE'),
           metrics.MeanAbsoluteError(name='MAE')
           ]
compile_optimizer = Adam(learning_rate=0.0001)

autoencoder.compile(loss='mse', 
                    optimizer=compile_optimizer,
                    metrics=METRICS)

            
image_gen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.15
    # preprocessing_function=AHE
    )

image_generator = image_gen.flow(
    x=x_images,
    y=y_images,
    batch_size=batch_size,
    shuffle=False)



ae_history = autoencoder.fit(image_generator,
                epochs=epochs,
                steps_per_epoch= len(x_images) // batch_size,
                batch_size=batch_size,
                validation_data=image_generator
                )



fig, ax = plt.subplots()

ax.plot(ae_history.history['loss'])
# ax.plot(history.history['val_loss'])
# ax.set_ylim(0)
ax.set_title('Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
# ax.legend(['train', 'val'], loc='best')


path = '/content/JSRT/JSRT/JPCLN036.png'
image = load_img(path,color_mode='grayscale', target_size=(img_rows,img_cols))
input_arr = img_to_array(image)
# input_arr = np.array([input_arr])
input_arr = input_arr
probs = autoencoder.predict(input_arr.reshape(-1, img_rows, img_cols, 1))

probs.shape


fig, ax = plt.subplots(1,2, figsize=(8,16))
ax[0].imshow(input_arr.reshape(img_rows, img_cols))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(probs.reshape(img_rows, img_cols))
ax[1].set_xticks([])
ax[1].set_yticks([])


