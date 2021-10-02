# KNN & KMean in Rust

> Dataset downloaded from http://yann.lecun.com/exdb/mnist/

> Original reference (it is in C++) : https://www.youtube.com/playlist?list=PL79n_WS-sPHKklEvOLiM1K94oJBsGnz71

NOTE: Before running the code, decompress the .gz files in `dataset/` directory

```sh
gunzip dataset/*
```

## What we are doing in KNN here

* Divide the subset 

In training we do these:
* Iterate over multiple values of k
* For each value of k
  * we iterate over 'all' points in validation set...
    * find k nearest points for each point
    * predict now (ie. check what is the most common label of k-nearest neighbours, that were just calculated)
    * if( predicted_label == correct_label ), we increase count_correct, and so performance is better

In this way, we can say... the validation set, was used to chose the best value of k

Now, to have a final metric on how accurate it 'may' be, we use the test dataset, just think of it same as the above steps, only difference is k is fixed now, we just want the performance value (ie. how many count_correct)

