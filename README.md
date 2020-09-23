# Chest X-Ray Classification And Shaddow Removal

Brent Thayer



## Topic
The topic of this project was to try to improve a network's ability to accurately predict a chest X-ray.  My main question is if removing the bones from the X-rays could possibly help.  I tried to accomplish this by the use of an autoencoder.

## Data

https://www.kaggle.com/hmchuong/xray-bone-shadow-supression

My dataset to accomplish this consisted of 241 pairs of images.

### X Data
<img src="/images/JPCLN001bone.png" alt="bone"
	title="no_bone" width="250" height="250" />

### Y Data (target)
<img src="/images/JPCLN001_1.png" alt="no_bone"
	title="no_bone" width="250" height="250" />

### Loss 
<img src="/images/first_500_oo_1500.png" alt="loss"
	title="loss" width="450" height="250" />

I observed my MSE dropping off fairly quickly.  

### Final Autoencoder Prediction

<img src="/images/val_loss_65_58.png" alt="pred"
	title="pred" width="500" height="250" />

## Model

<img src="/images/FINAL_MODEL.png" alt="final_model"
	title="final_model" width="600" height="400" />



[Presentation](https://github.com/brentthayer1/capstone3/tree/master/cs3_slides_final_pdf.pdf)