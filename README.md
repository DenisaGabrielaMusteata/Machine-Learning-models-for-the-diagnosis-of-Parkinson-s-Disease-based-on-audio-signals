# Parkinson's Disease Diagnosis Using Audio Signals

This project explores machine learning techniques for diagnosing **Parkinson’s disease** based on **audio signal analysis**, with a focus on speech impairments in prosody, articulation, and phonation. Two main approaches are developed and compared:

- A **Random Forest classifier** using **Fourier features**
- A **Convolutional Neural Network (AlexNet)** using **Mel-spectrograms**, optimized with a **Genetic Algorithm**

## Motivation

Parkinson's disease affects speech in non-trivial ways. By capturing and analyzing voice patterns, this project aims to create a **non-invasive diagnostic tool** using **supervised learning**.

## Technologies Used

- **MATLAB**
- **Machine Learning (Random Forest)**
- **Deep Learning (AlexNet CNN)**
- **Fourier Transform**
- **Mel-Spectrograms**
- **Genetic Algorithms**

## Dataset

- **Source:** [Zenodo - Parkinson’s Voice Dataset](https://zenodo.org/records/2867216)
- **Contents:** Audio recordings (read text and spontaneous conversation) from healthy and Parkinson’s patients.
- **Format:** `.wav`, 44.1kHz, 16-bit

## Methods

### 1. **Fourier + Random Forest**
- Preprocessing: Framing, filtering
- Feature extraction: Discrete Fourier Transform (`fft`)
- Classifier: Random Forest (tested with 50–200 trees)
- Result: ~80% test accuracy

### 2. **Mel-Spectrograms + CNN**
- Spectrograms created using variable frequency bands and window sizes
- Model: AlexNet CNN
- Optimization: Genetic Algorithm determines best parameters
- Best Result: >90% accuracy after GA optimization

## Genetic Algorithm Details

- Chromosome: [window size exponent, band count exponent]
- Fitness: CNN training accuracy
- Operators: Selection, Crossover (0.7), Mutation (0.1)
- Optimizes Mel-spectrogram configuration for better CNN performance

## Results

| Method         | Accuracy (Test) |
|----------------|-----------------|
| Random Forest  | ~80%            |
| CNN (AlexNet) + Mel | ~85–90% (optimized) |

## Future Work

- Apply genetic optimization to other CNN parameters
- Expand dataset with more diverse audio samples
- Integrate LSTM or hybrid architectures

## Author

**Denisa Gabriela Musteață**  
`denisa-gabriela.musteata@student.tuiasi.ro`  
Technical University “Gheorghe Asachi” of Iași  
Department of Automatics and Applied Informatics

---

> This repository demonstrates how voice-based biomarkers combined with machine learning can support early detection and monitoring of neurodegenerative disorders.
