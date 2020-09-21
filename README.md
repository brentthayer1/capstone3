# Capstone 3 Proposal

Brent Thayer



## Topic
The topic of this project was to try to improve a network's ability to accurately predict a chest X-ray.  My main question is if removing the bones from the X-rays could possibly help.  I tried to accomplish this by the use of a denoising autoencoder.

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

I observed my MSE dropping off fairly quickly.  I let my final model train for a total of 10,000 epochs in hopes the loss drop even more.  

### Final Autoencoder Prediction

<img src="/images/ae_prediction.png" alt="pred"
	title="pred" width="500" height="250" />

My output from the autoencoder was not what I expected, but continued on to test my classification model with and without the autoencoder.


### Result of network with AE

image

### Result without AE

image

## Final Model

<img src="/images/final_model_metrics.png" alt="final_model"
	title="final_model" width="500" height="250" />

### Final Model Predictions On Unseen Images

<img src="/images/cov_pred.png" alt="cov_pred"
	title="cov_pred" width="500" height="250" />

<img src="/images/pneum_pred.png" alt="pneum_pred"
	title="pneum_pred" width="500" height="250" />

<img src="/images/norm_pred.png" alt="norm_pred"
	title="norm_pred" width="500" height="250" />





To top it off though for my MVP, I would like to also encorporate a COVID class in addition to no finding and pneumonia.  Initially I would like to try and work with the architecture I built for the previous project since I got pretty good mileage out of it.  If this doesn't pan out, I would like to explore some transfer learning options.

