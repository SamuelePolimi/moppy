---
title: Deep Pro MP
created: 2024-11-02
last_updated: 2024-11-02
---

**Probabilistic Movement Primitives** (ProMPs) offer a powerful framework for robot
skill learning by representing movements as distributions of trajectories.

In this part of the library we use **neural network** to create Probabilistic Movement Primitives, thus the name Deep Pro MPs.
## Structure

<!--- pyreverse -o png --colorized -k moppy -->
![Uml structure of the library](/assets/img/deep-promp/classes.png)

The main part is the class **DeepProMP** it implements the core training algorithm. It used a DecoderDeepProMP and a EncoderDeepProMP.
