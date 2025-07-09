#  Parkinson's Disease Diagnosis Using Audio Signals (Bachelor's Degree Thesis)

This project was developed as part of the **Bachelor’s Degree thesis** for the completion of studies at the **Technical University “Gheorghe Asachi” of Iași**, Faculty of Automatic Control and Computer Engineering. It explores machine learning techniques for diagnosing **Parkinson’s disease** based on **audio signal analysis**, focusing on speech impairments in prosody, articulation, and phonation.

Two main approaches are developed and compared:

- A **Random Forest classifier** using **Fourier features**
- A **Convolutional Neural Network (AlexNet)** using **Mel-spectrograms**, optimized with a **Genetic Algorithm**

##  Motivation

Parkinson's disease significantly affects speech. By analyzing vocal signals, this project proposes a **non-invasive and efficient diagnostic method** using **supervised learning algorithms**.

##  Technologies Used

- **MATLAB**
- **Machine Learning (Random Forest)**
- **Deep Learning (AlexNet CNN)**
- **Fourier Transform**
- **Mel-Spectrograms**
- **Genetic Algorithms**

##  Dataset

- **Source:** [Zenodo - Parkinson’s Voice Dataset](https://zenodo.org/records/2867216)
- **Contents:** Audio recordings (reading + conversation) from Parkinson’s patients and healthy individuals
- **Format:** `.wav`, 44.1kHz, 16-bit

##  Methods

### 1. **Fourier Features + Random Forest**
- Preprocessing: Signal framing and filtering
- Feature extraction: Discrete Fourier Transform (`fft`)
- Classification: Random Forest with 50–200 trees
- **Accuracy:** ~80% on test data

### 2. **Mel-Spectrograms + CNN (AlexNet)**
- Input: Mel-spectrograms with tunable window size and frequency bands
- Model: Modified AlexNet CNN
- Optimization: **Genetic Algorithm** for spectrogram parameters
- **Accuracy:** >90% on test data after optimization

##  Genetic Algorithm Overview

- Chromosome: `[window_size_exponent, mel_band_exponent]`
- Fitness function: CNN training accuracy
- Parameters: Crossover = 0.7, Mutation = 0.1
- Used to find optimal Mel-spectrogram configuration

##  Results Summary

| Method              | Test Accuracy      |
|---------------------|--------------------|
| Random Forest (FFT) | ~80%               |
| CNN (AlexNet + Mel) | ~85–90% (optimized)|

##  Future Work

- Apply genetic optimization to additional CNN hyperparameters
- Augment the dataset with more varied and natural speech samples
- Experiment with RNNs/LSTM architectures for sequential analysis

##  Author

**Denisa Gabriela Musteață**  
`denisa-gabriela.musteata@student.tuiasi.ro`  
Technical University “Gheorghe Asachi” of Iași  
Department of Automatics and Applied Informatics

---

>  *This project was developed as a final year thesis for the completion of a Bachelor's Degree in Automation and Applied Informatics.*
>  
>  It showcases how machine learning can be effectively applied to non-invasive medical diagnosis based on voice biometrics.
> 
>  *For educational and non-commercial use only.*

