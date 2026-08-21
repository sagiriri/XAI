import numpy as np
import torch


def paper_style_deletion_auc(
    model,
    image_tensor,
    target_class,
    explanation,
    device,
    batch_size=None,
):
    """
    Paper-aligned batched deletion AUC.

    Lower deletion AUC indicates that important features are removed
    quickly according to the explanation ranking.

    This implementation uses:
      - explanation-ranked pixels
      - batched perturbation
      - black occlusion baseline
    """

    height, width = explanation.shape

    importance = np.abs(explanation).reshape(-1)
    pixel_order = np.argsort(-importance)

    total_pixels = height * width

    if batch_size is None:
        batch_size = max(height, width)

    batch_size = max(1, int(batch_size))

    baseline_value = 0.0

    def predict_probability(image):
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)
            return probabilities[0, target_class].item()

    masked = image_tensor.clone()

    fractions = [0.0]
    probabilities = [predict_probability(masked)]

    removed = 0

    while removed < total_pixels:
        end = min(removed + batch_size, total_pixels)

        selected = pixel_order[removed:end]

        rows = selected // width
        cols = selected % width

        masked[:, rows, cols] = baseline_value

        removed = end

        fractions.append(removed / total_pixels)
        probabilities.append(predict_probability(masked))

    return float(
        np.trapezoid(
            np.asarray(probabilities),
            np.asarray(fractions),
        )
    )
