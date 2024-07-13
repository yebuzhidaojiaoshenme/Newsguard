import matplotlib.pyplot as plt

def visualize_results(data, predictions):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):
        ax[i].imshow(data['image'][i].permute(1, 2, 0))
        ax[i].set_title(f"Prediction: {'Fake' if predictions[i] > 0.5 else 'Real'}")
        ax[i].axis('off')

    plt.show()
