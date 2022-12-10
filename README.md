# ECE251C
ECE251C Final project on speech enhancement

# Tasks
- Dataloader 
  - [ ] Saving subset of dataset for train-val-test (~10k)
  - [ ] Adding code to dataloader to load fixed length signal (~10sec)
  - [ ] Adding code to dataloader for passing real+imaginary spectrum for model 2
- Model
  - [ ] Adding pooling support
  - [ ] Adding wavelet pooling
- Evaluation
  - [ ] ?
- Experiments
  - [ ] (A) Baseline (magnitude only)
  - [ ] (B) Magnitude + phase
  - [ ] A and B with wavelet pooling
- Report / Presentaion
  - [ ] ?


# Experiment tracker
1. M1+M2+pooling on low res spectorogram (in PK private space) (50 train , 2 eval)
2. M1+M2+pooling on high res spectorogram (in PK private laptop) (25 train , 5 eval)
2. M1+M2+pooling on high res spectorogram (in common) (25 train , 5 eval) (adam) (running.....)

# Data pickle tracker
1. low res 6k train + 1k test (saqib laptop)
2. high res 1k + 0.1k test(teams) running....
3. normalized data (to do...)
