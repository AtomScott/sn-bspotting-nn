import numpy as np


def main():
    N = 5  # number of sequences
    T = 200  # timesteps in each sequence

    for split in ["train", "valid", "test"]:
        for i in range(N):
            # Randomly choose a sequence length 10% off from T
            t = np.random.randint(low=int(0.9 * T), high=int(1.1 * T))

            # Simulate the features and labels
            features = []
            labels = []

            for _ in range(t):
                # Randomly choose a class
                class_choice = np.random.choice(["Pass", "Drive", "None"])
                if class_choice == "Pass":
                    # Pass actions have lower values (with some overlap)
                    features.append(np.random.uniform(low=0.0, high=0.5, size=(784)))
                    labels.append(0)
                elif class_choice == "Drive":
                    # Drive actions have higher values (with some overlap)
                    features.append(np.random.uniform(low=0.5, high=1.0, size=(784)))
                    labels.append(1)
                else:
                    # None actions have middle-range values (with some overlap)
                    features.append(np.random.uniform(low=0.25, high=0.75, size=(784)))
                    labels.append(2)

            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)

            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, features.shape)  # mean=0, std dev=0.1
            features += noise

            # print the shapes of the features and labels
            print(f"features shape: {features.shape}")
            print(f"labels shape: {labels.shape}")

            # Save features and labels as separate .npy files
            np.save(f"sim_data/{split}/features_{i}.npy", features)
            np.save(f"sim_data/{split}/labels_{i}.npy", labels)


if __name__ == "__main__":
    main()
