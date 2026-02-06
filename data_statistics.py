import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "synthetic_data.csv"  # đổi thành data.csv nếu dùng data thật

def main():
    df = pd.read_csv(DATA_PATH)

    print("===== BASIC INFORMATION =====")
    print("Number of samples :", df.shape[0])
    print("Number of features:", df.shape[1] - 1)
    print("Columns:", list(df.columns))

    # ----------------------------
    # Label distribution
    # ----------------------------
    label_counts = df["label"].value_counts().sort_index()
    print("\n===== LABEL DISTRIBUTION =====")
    print(label_counts)

    label_counts.to_csv("label_distribution.csv")

    plt.figure()
    label_counts.plot(kind="bar")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.savefig("label_distribution.png")
    plt.show()

    # ----------------------------
    # Descriptive statistics
    # ----------------------------
    desc = df.describe()
    desc.to_csv("data_statistics.csv")
    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(desc)

    # ----------------------------
    # Example feature histograms
    # ----------------------------
    selected_features = [
        "age", "income", "risk_perception",
        "death_probability", "ticket_price"
    ]

    for col in selected_features:
        if col in df.columns:
            plt.figure()
            plt.hist(df[col], bins=30)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(f"{col}_distribution.png")
            plt.show()


if __name__ == "__main__":
    main()
