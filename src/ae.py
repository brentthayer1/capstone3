img_rows, img_cols = 300, 300

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    chanDim = 1
else:
    input_shape = (img_rows, img_cols, 1)
    chanDim = -1

epochs = 500
batch_size = 16

def images_resize(path, img_rows, img_cols):
  lst = [os.path.join(path, i) for i in os.listdir(path)]
  return [cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_rows, img_cols)) for path in lst]


image_dir = "/content/JSRT/JSRT/"
label_dir = "/content/BSE_JSRT/BSE_JSRT/"

x_data = images_resize(image_dir, img_rows, img_cols)
y_data = images_resize(label_dir, img_rows, img_cols)

x_images = np.array([i for i in x_data]).reshape(-1,img_rows,img_cols,1)
y_images = np.array([i for i in y_data]).reshape(-1,img_rows,img_cols,1)


input_img = Input(shape=input_shape)
# x = Conv2D(8, kernel_size=5, padding="SAME", activation="relu")(input_img)
x = Conv2D(16, kernel_size=5, padding="SAME", activation="relu")(input_img)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.10)(x)
x = Conv2D(32, kernel_size=5, padding="SAME", activation="relu")(x)
# x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.20)(x)
x = Conv2D(64, kernel_size=5, padding="SAME", activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.20)(x)
x = Conv2DTranspose(32, kernel_size=5, strides=2, padding="SAME", activation="relu")(x)
# x = Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="relu")(x)
x = Conv2D(64, kernel_size=5, padding="SAME", activation="relu")(x)
x = Conv2DTranspose(16, kernel_size=5, strides=2, padding="SAME", activation="relu")(x)

x = Conv2D(1, kernel_size=5, padding="SAME", activation="relu")(x)
autoencoder = Model(input_img, x)

autoencoder.summary()


name = '/content/gdrive/My Drive/TF_models/ae_model300_4.h5'
mc = ModelCheckpoint(name, monitor='val_loss', 
                     mode='min', verbose=1, save_best_only=True)





METRICS = [metrics.RootMeanSquaredError(name='RMSE'),
           metrics.MeanAbsoluteError(name='MAE')
           ]
compile_optimizer = Adam(learning_rate=0.000001)

autoencoder.compile(loss='mse', 
                    optimizer=compile_optimizer,
                    metrics=METRICS)


image_gen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.25
    )

image_generator = image_gen.flow(
    x=x_images,
    y=y_images,
    batch_size=batch_size,
    shuffle=True,)


autoencoder.summary()


ae_history = autoencoder.fit(image_generator,
                epochs=epochs,
                steps_per_epoch= len(x_images) // batch_size,
                batch_size=batch_size,
                validation_data=image_generator,
                callbacks=[mc]
                )




path = '/content/JSRT/JSRT/JPCLN018.png'
image = load_img(path,color_mode='grayscale', target_size=(img_rows,img_cols))
input_arr = img_to_array(image).reshape(img_rows, img_cols)
result = autoencoder.predict(input_arr.reshape(1, img_rows, img_cols, 1)).reshape(img_rows, img_cols)  

fig, ax = plt.subplots(1, 2, figsize=(10,20))

ax[0].imshow(input_arr)
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)

ax[1].imshow(result)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)



fig, ax = plt.subplots()

ax.plot(ae_history.history['loss'])
# ax.plot(history.history['val_loss'])
# ax.set_ylim(0)
ax.set_title('Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
# ax.legend(['train', 'val'], loc='best')


