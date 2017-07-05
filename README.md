# tf-face-clustering  
Implementation of face clustering.  
Instead of using fine tune, I used an unsupervised auto-encoder trained on CelebA.  
The clustering network is based on center-loss and trained with Muct which only have 3000+ faces given by 200+ people.
  
All of the data is aligned with OpenCV 'lbpcascade_frontalface', the wrong detections are simply dismissed.  
  
  
---  
  
### Convolutional AutoEncoding Network  
  
  
---  
  
### Clustering Network  
  