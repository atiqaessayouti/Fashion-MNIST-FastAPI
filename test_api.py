"""Test de l'API sur 20 images Fashion-MNIST + test d'une entrée incorrecte.

Lancer le serveur dans un premier terminal :
    uvicorn app:app --reload

Puis, dans un second terminal :
    python test_api.py
"""
from pathlib import Path
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CAPTURES_DIR = BASE_DIR / "captures"
RESULTS_DIR.mkdir(exist_ok=True)
CAPTURES_DIR.mkdir(exist_ok=True)

API_URL = "http://127.0.0.1:8000"
N_IMAGES = 20
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def main():
    # Vérifier que l'API est démarrée.
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        health.raise_for_status()
        print("Health :", health.json())
    except requests.RequestException as exc:
        raise SystemExit(
            f"API inaccessible ({exc}). Lancez d'abord : uvicorn app:app --reload"
        )

    # Charger le jeu de test officiel Fashion-MNIST.
    (_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    rows = []
    tested_images = []

    for i in range(N_IMAGES):
        image = test_images[i]
        payload = {"pixels": image.astype(np.float32).reshape(-1).tolist()}

        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        true_id = int(test_labels[i])
        predicted_id = int(result["class_id"])
        correct = true_id == predicted_id

        rows.append({
            "image": i,
            "classe_reelle": CLASS_NAMES[true_id],
            "classe_predite": result["class_name"],
            "confiance": result["confidence"],
            "latence_ms": result["latency_ms"],
            "resultat": "Correct" if correct else "Incorrect",
        })
        tested_images.append(image)
        print(
            f"{i:02d} | réelle={CLASS_NAMES[true_id]:12s} | "
            f"prédite={result['class_name']:12s} | "
            f"confiance={result['confidence']:.4f} | "
            f"latence={result['latency_ms']:.3f} ms | "
            f"{'Correct' if correct else 'Incorrect'}"
        )

    correct_count = sum(row["resultat"] == "Correct" for row in rows)
    accuracy = correct_count / N_IMAGES * 100.0
    mean_latency = float(np.mean([row["latence_ms"] for row in rows]))

    # Export CSV des 20 prédictions.
    csv_path = RESULTS_DIR / "predictions_20.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "nombre_images": N_IMAGES,
        "predictions_correctes": correct_count,
        "taux_reussite_pct": round(accuracy, 2),
        "latence_moyenne_ms": round(mean_latency, 3),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Figure des 20 images réellement envoyées à l'API.
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(tested_images[i], cmap="gray")
        row = rows[i]
        ax.set_title(
            f"{i}: {row['classe_predite']}\n"
            f"Réel: {row['classe_reelle']} | {row['resultat']}",
            fontsize=7,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "images_testees.png", dpi=150)
    plt.close(fig)

    # Test négatif : seulement trois pixels.
    bad_response = requests.post(
        f"{API_URL}/predict",
        json={"pixels": [0, 128, 255]},
        timeout=10,
    )
    bad_result = {
        "status_code": bad_response.status_code,
        "response": bad_response.json(),
    }
    (RESULTS_DIR / "test_entree_incorrecte.json").write_text(
        json.dumps(bad_result, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== BILAN =====")
    print(f"Prédictions correctes : {correct_count}/{N_IMAGES}")
    print(f"Taux de réussite : {accuracy:.2f}%")
    print(f"Latence moyenne : {mean_latency:.3f} ms")
    print("\nTest incorrect :", bad_response.status_code, bad_response.json())
    print(f"CSV créé : {csv_path}")
    print("Graphique créé :", RESULTS_DIR / "images_testees.png")

    # Générer le compte rendu avec les résultats réellement mesurés.
    try:
        from generate_report import generate_report
        generate_report(rows, summary, bad_result)
        print("Compte rendu créé :", BASE_DIR / "compte_rendu.pdf")
    except Exception as exc:
        print("Attention : génération PDF non effectuée :", exc)

if __name__ == "__main__":
    main()
