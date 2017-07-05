# tf-face-clustering  
Implementation of face clustering.  
Instead of using fine tune, I trained an unsupervised auto-encoder on CelebA.  
The clustering network is based on center-loss and trained with Muct which only have 3000+ faces given by 200+ people.
  
All of the data is aligned with OpenCV 'lbpcascade_frontalface', the wrong detections are simply dismissed.  
  
---  
  
### Embedding Graph  
<img src="https://github.com/htkseason/tf-face-clustering/blob/master/demo/embedding_graph_01.png" width="75%"/>  
<img src="https://github.com/htkseason/tf-face-clustering/blob/master/demo/embedding_graph_02.png" width="75%"/>  
  
---  
  
### Convolutional AutoEncoding Network  
<img src="https://github.com/htkseason/tf-face-clustering/blob/master/demo/enet_loss.png" width="75%"/>  
  
---  
  
### Clustering Network  
<img src="https://github.com/htkseason/tf-face-clustering/blob/master/demo/cnet_softmax_loss.png" width="75%"/>  
<img src="https://github.com/htkseason/tf-face-clustering/blob/master/demo/cnet_center_loss.png" width="75%"/>  