# Anomaly Detection in Surveillance Videos using Machine Learning
  Our Anomaly detection system for surveillance monitoring aims to capture a diverse range of realistic anomalies in surveillance videos. Leveraging visual technologies, our proposed method detects various nomalies including theft, aggression, and accidents. By integrating normal and anomalous behavior data, our approach facilitates prompt identification and real-time responses to suspicious activities, enhancing security and public safety. Integrated with surveillance cameras in public spaces, the system alerts security personnel to instances of suspicious activity, ensuring timely intervention. We employ a deep anomaly ranking model to autonomously predict high anomaly scores within anomalous videos, thereby improving detection accuracy. Additionally, our method enables concurrent detection of anomalous activity in video frames and triggers an email alert mechanism with a photo of the detected anomaly, thus advancing anomaly detection capabilities and proactive security management in public spaces.

  ## Learning Model

  ![image](https://github.com/user-attachments/assets/dffaa881-889b-45dd-9f8c-26958866c54b)


# Technologies Used
  * Machine Learning
  * Computer Vision
  * C3D Model
  * LSTM(Long Short-Term Memory)
  * SMTP(Simple Mail Transfer Protocol)
  * Twilio

# Modules
# Data Collection
  The process of data collection from the UCF Crime Dataset involves downloading video clips depicting various criminal and non-criminal activities, annotating them with relevant labels, and splitting the dataset into training and testing sets. Preprocessing steps, including resizing frames and normalizing pixel values, ensure data consistency, while optional augmentation techniques enhance diversity. Features are extracted from video frames, often utilizing pre-trained convolutional neural networks, and a deep anomaly ranking model is trained on the annotated dataset. The model is fine-tuned and evaluated on the testing set, using metrics like precision and recall to gauge its effectiveness in detecting criminal activities. This comprehensive approach ensures the dataset's utilization for training and evaluating anomaly detection models in surveillance videos from the UCF Crime Dataset.
  
   ## UCF Crime Dataset(Normal frames)
  
  ![image](https://github.com/user-attachments/assets/40f102c6-2497-4455-bc95-d7650a53a633)

  ## UCF Crime Dataset(Anomalous frames)
  
  ![image](https://github.com/user-attachments/assets/5eb5ab7b-56e0-4b91-b0b8-cddf16492ad1)

  ## Dataset Collection

  ![image](https://github.com/user-attachments/assets/bf47b940-99cc-4488-9dbb-7e5cea5c59f1)




# Model Training
  In order to train a deep anomaly ranking model for efficient anomaly detection in surveillance footage, the annotated and preprocessed UCF Crime Dataset is utilized. The model is trained to identify patterns suggestive of criminal activity using a combination of deep multiple instance ranking framework and weakly labeled training films, where video-level annotations take the place of detailed segment annotations. In order to improve anomaly localization during the learning process, sparsity and temporal smoothness restrictions are added to the ranking loss function during the training phase. The trained model learns to prioritize anomaly detection while preserving temporal consistency by automatically assigning high anomaly scores to video segments. This approach, which is incorporated into the model-training procedure, guarantees that the system can detect abnormalities and improve security and situational awareness in surveillance settings that make use of the UCF Crime Dataset.

  ##  CNN & LSTM Model

  ![image](https://github.com/user-attachments/assets/a9ca7757-eb0d-40ec-864e-7c8def9bc276)

  ## Neural Network Evaluation

  ![image](https://github.com/user-attachments/assets/55d5b064-31de-4ca8-8a33-ef707b2face7)



# Anomaly Detection and Testing
  Our technique includes a set of methodical measures to guarantee the effectiveness and dependability of the anomaly detection system for the surveillance footage anomaly detection project. To make model raining and testing easier, we first carefully chose a suitable dataset, giving preference to those containing labeled anomalies, such the UCFCrime dataset. We carefully preprocessed the surveillance footage after choosing the dataset, doing operations like frame extraction, resizing, normalization, and augmentation to standardize the data and enhance feature extraction. After the data was ready, we chose appropriate deep learning architectures taking into account the expected kinds of abnormalities and the intricacy of the surveillance video. To find the best model for anomaly detection, we investigated a variety of rchitectures, such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and combinations like Convolutional LSTM (ConvLSTM) networks. We then started the training phase, during which the chosen model was trained to differentiate between patterns of normal and aberrant behavior seen in the surveillance footage. Our goal was to minimize false positives and improve the model's accuracy in detecting abnormalities through iterative training and improvement. Using a different testing dataset, thorough testing and evaluation were carried out once the model had been trained. To evaluate the model's performance in a comprehensive way, we used well-established evaluation metrics such F1 score, precision, recall, and Receiver Operating Characteristic (ROC) curves.

  ## Anomaly detection

  ![image](https://github.com/user-attachments/assets/fe1aa451-2e4a-48f8-9022-cd9fd752b098)


# Alerting Personnel
  This action was taken to guarantee a prompt and efficient reaction to anomalies that were found. First, the Alerting Personnel module is triggered to start the alerting process when the anomaly detection system finds anomalies in the surveillance footage. This module is made to quickly produce warnings and inform authorities or designated security personnel about abnormalities that are found. This notification system makes sure that staff members are notified as soon as possible about possible security breaches or threats by using several communication channels including email, SMS, or specialized alerting systems. Alerts can be prioritized and escalated as needed because the alerting process is tailored to the specifics of the abnormalities identified. As an instance, notifications about critical security issues could prompt quick action, whereas notifications about anomalies of lesser priority could be forwarded for more investigation or observation

  ## Alerting personnel (Email)

  ![image](https://github.com/user-attachments/assets/33a71acd-b765-4c6a-91f5-20bde366626e)

  ## Alerting personnel (SMS)

  ![image](https://github.com/user-attachments/assets/4a49fd2e-b7ba-466f-93c1-6d7666b55de7)


  
