# MADDQN-based Subnetwork Channel Allocation  
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021b%2B-blue.svg)](https://www.mathworks.com/)  
[![Paper](https://img.shields.io/badge/WCNC-2023-brightgreen)](MAQL_for_Resource_Allocation.pdf)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

MATLAB implementation of the WCNC 2023 paper **“Distributed Channel Allocation for Mobile 6G Subnetworks via Multi-Agent Deep Q-Learning.”**  

---

## 📌 Table of Contents  

- [Motivation](#motivation)  
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Experiments & Results](#experiments--results)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## 🚀 Motivation  

Future **6G mobile networks** require efficient subnetwork partitioning and distributed channel allocation to maximize **spectral efficiency**, **reduce interference**, and **ensure QoS**.  

This project leverages **Multi-Agent Deep Q-Networks (MADDQN)** to solve the **distributed channel allocation problem** in subnetworks, as described in the referenced WCNC 2023 paper.  

---

## ✨ Features  

- 🧠 Multi-Agent DQN for decentralized channel allocation  
- 📊 MATLAB implementation with reusable components  
- 🎯 Reward / objective function customizable for different scenarios  
- 🔄 Experience replay memory support  
- 📉 Visualization tools for training and performance metrics  

---

## 📂 Repository Structure  

| File / Folder | Description |
|---------------|-------------|
| `RLPowerControlDQNC.m` | Main MATLAB script for running the RL algorithm |
| `subnetwork_classC.m` | Subnetwork agent / environment modeling |
| `storeExperienceM.m` | Experience replay memory handler |
| `objective_func.m` | Reward and objective function definitions |
| `hBuildFigure.m` | Utility for result plotting |
| `MAQL_for_Resource_Allocation.pdf` | WCNC 2023 paper reference |
| `README.md` | Project documentation (this file) |

---

## 🔧 Requirements  

- **MATLAB R2021b+** (earlier versions may also work)  
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)  
- [Reinforcement Learning Toolbox](https://www.mathworks.com/products/reinforcement-learning.html)  

---

## ⚙️ Installation  

Clone the repository and open in MATLAB:  

```bash
git clone https://github.com/ramonadeog/MADDQN-based-subnetwork-channel-allocation.git
cd MADDQN-based-subnetwork-channel-allocation
```
---

## ⚙️ Configuration  

You can tune the following parameters in the main script or config sections:  

- **Number of subnetworks / agents**  
- **Number of available channels**  
- **Reward shaping parameters**  
- **Learning rate, discount factor, ε-greedy schedule**  
- **Training episodes / evaluation episodes**  

---

## 📊 Experiments & Results  

The implementation can reproduce results from the WCNC 2023 paper:  

- **Throughput vs. number of subnetworks**  
- **Interference and spectral efficiency metrics**  
- **Convergence curves of training episodes**  

Figures are generated via `hBuildFigure.m`.  

---

## 🤝 Contributing  

Contributions are welcome! You can help by:  

- Fixing bugs and improving code efficiency  
- Extending to other interference / channel models  
- Adding new visualizations or logging utilities  
- Porting to Python (e.g., with PyTorch / TensorFlow)  

Please open an **issue** or submit a **pull request**.  

---

## 📜 License  

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  

---

## 📧 Contact  

- GitHub: [@ramonadeog](https://github.com/ramonadeog)  
- For academic references, cite the WCNC 2023 paper included in the repo.  

