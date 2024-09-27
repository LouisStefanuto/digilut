# Post processing and submission generation

!!! warning
    The postprocessing is not implemented yet.

    I conducted some experiments but this part of the pipeline is still **work in progress**.

With the patch classification pipeline, we get coarse segmentation maps of the slide.

The DigiLut challenge required to predict bounding boxes. Thus, I want to add a postprocessing step to convert the masks into $N$ bounding boxes (the $N$ is given by the organizers for each slide).

I want to give a try to 2 approaches:

1. A clustering one, based on DBSCAN
2. A union-fuse one

## DBSCAN

- To extract bounding boxes, we first assign them to clusters using DBSCAN.
    - DBSCAN has the advantage of classifing outliers in a "-1" cluster, this is useful to avoid huge bounding boxes at the next steps, as we take englobing bboxes
- Finally we compute the englobing bounding boxes for each cluster.
- The submission file tells us how many N boxes are expected. So we only keep the N clusters with the most positive patches. If the number of predicted boxes is smaller than N, we pad with (0,0,0,0) predictions.

**Drawback**: we need to find the optimal clusters parameters (minimal number of points, distance for points to be linked), that are image-dependent.

## Union-fuse

This technique is simpler and it requires no hyperparameter tuning.

- Fuse overlapping positive patches until no one overlaps.
- Remove all positive patches that don't have at least two positive neighbors (within their 8 neighbors). That removes most of the outliers/false positive/not dense clusters.
- Compute the englobing bounding boxes for each cluster.
- The submission file tells us how many N boxes are expected. So we only keep the N clusters with the most positive patches. If the number of predicted boxes is smaller than N, we pad with (0,0,0,0) predictions.
