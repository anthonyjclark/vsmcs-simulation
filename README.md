# VSMCs Simulation

![](example-run.png)

In the image above: (top left) the original image; (top middle) the image binarized to {0 1}; (top right) a depiction of the connected components of the image; (bottom left) the lattice image where brightness denotes the number of connected components within range of the cell; (bottom middle) regions scored by the number of components and the brightness of the original image (top left); and (bottom right) the location of all simulated VSMCs.

# Files

```
├── README.md
├── example-run.png
├── images
│   └── binary.jpg
├── notes
│   ├── algorithm.jpg
│   ├── parameters.jpg
│   ├── simulation1.jpg
│   ├── simulation2.jpg
│   └── simulation3.jpg
└── vsmcs-simulation.ipynb
```

The `images` directory will contain images showing veins and capillaries without the VSMCs present.

The `notes` directory contains meeting notes.

# Notes

- compare with and without repulsion signal
