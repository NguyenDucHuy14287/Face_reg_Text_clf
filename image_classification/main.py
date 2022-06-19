from class_image_classify_model import *
import os
import pandas as pd

train_label_raw = pd.read_csv('df_train_label.csv', usecols=['label', 'image_path'])
total_row = train_label_raw.shape[0]

start_idx = 0
end_idx = 4000
num_ep = 10

trainer = image_binary_classify_keras()
trainer.init_model()

train_label_df = train_label_raw.iloc[start_idx:end_idx]
trainer.generate_train_valid(img_path_arr=train_label_df['image_path'], label_arr=train_label_df['label'])
trainer.train_model(num_epochs=num_ep)

# save model
os.makedirs('model_checkpoint', exist_ok=True)
trainer.save_model(f'model_checkpoint/checkpoint_{start_idx}_{end_idx}.h5')
