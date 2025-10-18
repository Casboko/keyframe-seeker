import faiss
import open_clip
import torch
from PIL import Image


def build_dummy_images(preprocess) -> torch.Tensor:
    image_a = Image.new("RGB", (224, 224), (127, 127, 127))
    image_b = Image.new("RGB", (224, 224), (160, 160, 160))
    batch = torch.stack(
        [
            preprocess(image_a),
            preprocess(image_b),
        ]
    )
    return batch


def main() -> None:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()

    images = build_dummy_images(preprocess)

    with torch.inference_mode():
        features = model.encode_image(images).float()

    vectors = features.cpu().numpy()
    faiss.normalize_L2(vectors)

    base_index = faiss.IndexFlatIP(vectors.shape[1])
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(base_index)
    else:
        index = base_index

    index.add(vectors)
    distances, indices = index.search(vectors[:1], 5)

    print("query_top5_indices:", indices[0].tolist())
    print("query_top5_distances:", distances[0].tolist())


if __name__ == "__main__":
    main()
