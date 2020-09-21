# Capstone 3 Proposal

Brent Thayer



## Tipic
The topic of this project was to try to improve a network's ability to accurately predict a chest X-ray.  My main question is if removing the bones from the X-rays could possibly help.  I tried to accomplish this by the use of a denoising autoencoder.

My dataset to accomplish this consisted of 241 pairs of images.
### X Data
<img src="/images/JPCLN001bone.png" alt="bone"
	title="no_bone" width="250" height="250" />

### Y Data (target)
<img src="/images/JPCLN001_1.png" alt="no_bone"
	title="no_bone" width="250" height="250" />



To top it off though for my MVP, I would like to also encorporate a COVID class in addition to no finding and pneumonia.  Initially I would like to try and work with the architecture I built for the previous project since I got pretty good mileage out of it.  If this doesn't pan out, I would like to explore some transfer learning options.

