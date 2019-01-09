import numpy as np
from data import *
from model import *

data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05,
                     height_shift_range=0.05, shear_range=0.05, 
                     zoom_range=0.05, horizontal_flip=True,
                     fill_mode='nearest')

traingen = train_gen(2,'data/train','images','labels',data_gen_args,seed=14832523)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(traingen, steps_per_epoch=300, epochs=3, callbacks=[model_checkpoint])

testGene = test_gen("data/test/images")
results = model.predict_generator(testGene, 30, verbose=1)
save_preds("data/test/preds2",results)