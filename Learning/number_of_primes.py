import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pmlb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math


# --- 1. Primality Test Helper Function ---
# A simple function to check if a number is prime.
# We'll use this to count primes in the model's weights.
def is_prime(n):
    """Checks if an integer is a prime number."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# --- 2. MLP Model Definition ---
# A simple two-layer MLP with BatchNorm and Dropout.
# It includes a custom method to count prime numbers in its weights.
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.fc1(x)
        # BatchNorm is applied before activation
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_primes(self, scaling_factor=1.0):
        """
        Counts the total and distinct prime numbers in the weights of the MLP.
        Rounds weights to the nearest integer after scaling before checking for primality.

        Args:
            scaling_factor (float): A factor to scale weights by before rounding.
                                    This helps in mapping small weight values to a
                                    larger integer range for more meaningful prime counting.
        """
        prime_counts = {}
        all_weights_int = []

        # --- Layer 1 ---
        weights_l1 = self.fc1.weight.detach().cpu()
        # Scale and round weights to get integers for primality testing
        weights_l1_int = (
            torch.round(weights_l1 * scaling_factor).int().flatten().tolist()
        )
        primes_l1 = [p for p in weights_l1_int if is_prime(abs(p))]
        prime_counts["layer1_primes"] = len(primes_l1)
        prime_counts["layer1_distinct_primes"] = len(set(primes_l1))
        all_weights_int.extend(weights_l1_int)

        # --- Layer 2 ---
        weights_l2 = self.fc2.weight.detach().cpu()
        # Scale and round weights
        weights_l2_int = (
            torch.round(weights_l2 * scaling_factor).int().flatten().tolist()
        )
        primes_l2 = [p for p in weights_l2_int if is_prime(abs(p))]
        prime_counts["layer2_primes"] = len(primes_l2)
        prime_counts["layer2_distinct_primes"] = len(set(primes_l2))
        all_weights_int.extend(weights_l2_int)

        # --- Total ---
        total_primes_list = [p for p in all_weights_int if is_prime(abs(p))]
        prime_counts["total_primes"] = len(total_primes_list)
        prime_counts["total_distinct_primes"] = len(set(total_primes_list))

        return prime_counts


# --- 3. Plotting Function ---
def plot_metrics(df, columns_to_plot, dataset_name="dataset"):
    """
    Plots specified metrics from the training history DataFrame.
    Uses a dual-axis plot for losses and prime counts.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Colors and markers for different lines
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "p"]

    # Create a second y-axis for prime counts
    ax2 = ax1.twinx()

    ax1_lines, ax2_lines = [], []
    ax1_labels, ax2_labels = [], []

    # Plot each specified column on the appropriate axis
    for i, col in enumerate(columns_to_plot):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        if "loss" in col:
            (line,) = ax1.plot(
                df["epoch"],
                df[col],
                label=col,
                color=color,
                marker=marker,
                linestyle="-",
                markersize=5,
            )
            ax1_lines.append(line)
            ax1_labels.append(col)
        else:
            (line,) = ax2.plot(
                df["epoch"],
                df[col],
                label=col,
                color=color,
                marker=marker,
                linestyle="--",
                markersize=5,
            )
            ax2_lines.append(line)
            ax2_labels.append(col)

    # --- Formatting the plot ---
    # Set axis labels
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color=colors[0])
    ax2.set_ylabel("Prime Counts", fontsize=14, color=colors[1])

    # Set tick colors
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    # Set title and grid
    plt.title("Training Metrics vs. Epochs", fontsize=16)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Combine legends from both axes
    lines = ax1_lines + ax2_lines
    labels = ax1_labels + ax2_labels
    ax1.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=len(labels),
    )

    fig.tight_layout()
    plt.savefig(f"./results/train_test_{dataset_name}.png")
    plt.show()


# --- 4. Main Training and Evaluation Script ---
def main():
    # --- Configuration ---
    DATASET_NAME = "Hill_Valley_with_noise"  # You can change this to any pmlb dataset
    TEST_SPLIT = 0.2
    BATCH_SIZE = 256
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 300
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.2
    WEIGHT_SCALING_FACTOR = 1000.0  # Factor to scale weights by for prime analysis

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    print(f"Loading dataset: {DATASET_NAME}...")
    X, y = pmlb.fetch_data(DATASET_NAME, return_X_y=True, local_cache_dir="./pmlb_data")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)  # Use LongTensor for CrossEntropyLoss

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=TEST_SPLIT, random_state=42, stratify=y
    )

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_tensor))

    model = SimpleMLP(input_dim, HIDDEN_DIM, output_dim, DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    history = []
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        # Evaluation phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)

        test_loss = running_test_loss / len(test_loader.dataset)

        # Count primes in model weights using the scaling factor
        prime_stats = model.count_primes(scaling_factor=WEIGHT_SCALING_FACTOR)

        # Record metrics for this epoch
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            **prime_stats,
        }
        history.append(epoch_stats)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Total Primes: {prime_stats['total_primes']} | "
            f"Distinct Primes: {prime_stats['total_distinct_primes']}"
        )

    # --- Results and Visualization ---
    history_df = pd.DataFrame(history)
    print("\nTraining complete. Final metrics:")
    print(history_df.tail())

    # Plot the results
    plot_metrics(
        history_df,
        columns_to_plot=[
            "train_loss",
            "test_loss",
            "total_primes",
            "total_distinct_primes",
        ],
        dataset_name=DATASET_NAME,
    )


if __name__ == "__main__":
    main()
