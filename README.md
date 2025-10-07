# Images to sketches using a half Cycle-Gan

This method takes a grayscale image as input and generates a stylized sketch version as output. The approach relies on a **half-CycleGAN architecture** (G → F → G with a single discriminator evaluating the generated sketches). This simplified architecture has been shown in the literature to offer a good trade-off between **computational cost and performance** compared to a full CycleGAN. The current generator is structured as an **encoder–decoder with ResNet blocks** and **attention-based skip connections**. The discriminator is based on the **PatchGAN** architecture.
The **loss function composition is still under investigation**. Initial experiments include standard losses (discriminator loss, identity loss), along with **SSIM loss** to promote structural consistency in the generated sketches.

This work is a work in progress
# Train
Execute training.py
# Test
Coming soon
# Results
Coming soon
# To Do
### Next steps to explore:
- Using color images as input
- Tuning hyperparameters (currently: `identity_loss = 0.5`, `cycle_loss = 15.0`)
- Integrating additional loss terms, such as **Canny** or **Sobel** based edge losses
