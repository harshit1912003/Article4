## 1. Data Generation in Iterations

For this analysis, we consider a dataset of 150 DMUs, each with three inputs: X₁, X₂, and X₃.  
- DMUs 1–100 produce three outputs: Y₁, Y₂, Y₃.  
- DMUs 101–150 produce only one output: Y₁ (non-homogeneous).  

All values are drawn uniformly from the finite set  
A = { -50, -49, …, -1, 1, 2, …, 50 }.

---

## 2. Data Generation in IterationsNORMAL

Same 150 DMUs and I/O structure. Each entry is sampled from a two-component normal mixture:

- With probability 0.5:  
  – Distribution: Normal(mean = 25, variance = 8.3³)  
- With probability 0.5:  
  – Distribution: Normal(mean = –25, variance = 8.3²)  

- DMUs 1–100 use features [X₁, X₂, X₃, Y₁, Y₂, Y₃]  
- DMUs 101–150 use features [X₁, X₂, X₃, Y₁]  
