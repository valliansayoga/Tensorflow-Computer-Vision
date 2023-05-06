import matplotlib.pyplot as plt

def display_some_examples(images, labels):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        idx = np.random.randint(0, images.shape[0] - 1)
        img = images[idx]
        label = labels[idx]

        plt.subplot(5, 5, i + 1)
        plt.title(str(label))  # type:ignore
        plt.tight_layout()
        plt.imshow(img, cmap="gray")
    plt.show()