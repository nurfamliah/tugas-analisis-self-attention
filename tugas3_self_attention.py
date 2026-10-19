import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)


def softmax(x, axis=-1):
    """Mengubah skor menjadi bobot probabilitas secara stabil."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Menghitung Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) V."""
    d_k = Q.shape[-1]
    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_k)

    if mask is not None:
        scaled_scores = np.where(mask == 1, -1e9, scaled_scores)

    weights = softmax(scaled_scores, axis=-1)
    output = weights @ V
    return output, weights


def create_causal_mask(seq_len):
    """Mencegah token melihat token-token di masa depan."""
    return np.triu(np.ones((seq_len, seq_len)), k=1)


def run_sentence_experiment():
    sentence = "Kucing kecil mengejar bola merah di halaman rumah."
    tokens = ["Kucing", "kecil", "mengejar", "bola", "merah", "di", "halaman", "rumah"]
    seq_len = len(tokens)
    d_model = 8

    # Representasi sintetis untuk demonstrasi mekanisme attention.
    # Ini bukan embedding bahasa Indonesia hasil pelatihan model.
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 0.5, size=(seq_len, d_model))
    W_q = rng.normal(0, 0.35, size=(d_model, d_model))
    W_k = rng.normal(0, 0.35, size=(d_model, d_model))
    W_v = rng.normal(0, 0.35, size=(d_model, d_model))

    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v

    output, attention = scaled_dot_product_attention(Q, K, V)

    # Cari bobot terbesar selain diagonal (agar fokus pada relasi antarkata berbeda).
    attention_off_diagonal = attention.copy()
    np.fill_diagonal(attention_off_diagonal, -1.0)
    q_idx, k_idx = np.unravel_index(np.argmax(attention_off_diagonal), attention_off_diagonal.shape)
    max_weight = attention_off_diagonal[q_idx, k_idx]

    print("=" * 68)
    print("EKSPERIMEN SELF-ATTENTION")
    print("Kalimat:", sentence)
    print("Jumlah kata:", seq_len)
    print("\nMatriks bobot attention (baris=Query, kolom=Key):")
    print(np.array2string(attention, precision=4, suppress_small=True))
    print(
        f"\nBobot antarkata terbesar: Query '{tokens[q_idx]}' -> "
        f"Key '{tokens[k_idx]}' = {max_weight:.4f} ({max_weight*100:.2f}%)"
    )
    print("Catatan: bobot ini berasal dari representasi sintetis acak, bukan model bahasa terlatih.")
    print("Bentuk output kontekstual:", output.shape)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
    image = ax.imshow(attention, cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Bobot attention")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Key Tokens (kata yang diperhatikan)")
    ax.set_ylabel("Query Tokens (kata yang sedang memperhatikan)")
    ax.set_title("Heatmap Self-Attention\nKucing kecil mengejar bola merah di halaman rumah.")
    for i in range(seq_len):
        for j in range(seq_len):
            ax.text(j, i, f"{attention[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if attention[i, j] < 0.45 else "black")
    fig.tight_layout()
    output_image = "attention_heatmap_kucing.png"
    fig.savefig(output_image, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap berhasil disimpan: {output_image}")
    return tokens, attention, (tokens[q_idx], tokens[k_idx], float(max_weight))


def demo_causal_mask():
    tokens = ["Saya", "sedang", "belajar", "AI"]
    mask = create_causal_mask(len(tokens))
    scores = np.zeros((len(tokens), len(tokens)))
    masked_scores = np.where(mask == 1, -1e9, scores)
    weights = softmax(masked_scores, axis=-1)
    print("\nContoh causal mask (1 = dilarang melihat token masa depan):")
    print(mask.astype(int))
    print("Bobot setelah mask (bagian kanan diagonal menjadi 0):")
    print(np.array2string(weights, precision=2, suppress_small=True))


if __name__ == "__main__":
    run_sentence_experiment()
    demo_causal_mask()
